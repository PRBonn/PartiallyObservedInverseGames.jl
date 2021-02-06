module InverseGames

import JuMP
import Ipopt
import ..DynamicsModelInterface
import ..JuMPUtils
import ..CostUtils
import ..InverseOptimalControl

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export InverseIBRSolver, InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game

#=========================================== Inverse IBR ===========================================#

struct InverseIBRSolver end

# TODO: allow for partial and noisy state observations
# TODO: This probably does not work so well if the initial strategy for the players does not match
# the observation well.
function solve_inverse_game(
    ::InverseIBRSolver,
    x̂;
    observation_model,
    control_system,
    player_cost_models,
    # TODO: dirty hack
    u_init = nothing,
    inner_solver_kwargs = (),
    max_ibr_rounds = 5,
    ibr_convergence_tolerance = 0.01,
    verbose = false,
)
    @unpack n_controls = control_system
    n_players = length(player_cost_models)

    last_ibr_solution = (; x = x̂, u = u_init, λ = nothing)
    last_player_solution = last_ibr_solution
    player_opt_models = resize!(JuMP.Model[], n_players)
    # TODO: dirty hack
    player_weights = Any[nothing for _ in 1:n_players]
    converged = false

    for i_ibr in 1:max_ibr_rounds
        for (player_idx, player_cost_model) in enumerate(player_cost_models)
            cost_model = player_cost_models[player_idx]
            last_player_solution, player_opt_models[player_idx] =
                InverseOptimalControl.solve_inverse_optimal_control(
                    x̂;
                    control_system,
                    cost_model,
                    observation_model,
                    fixed_inputs = filter(i -> i ∉ player_cost_model.player_inputs, 1:n_controls),
                    init = (;
                        weights = player_weights[player_idx],
                        # TODO: think about whether we should initialize with `x̂` or with
                        # `last_player_solution.x` here
                        # TODO: think about whether we should also hand over λ or whether it is
                        # better to start from λ = 0 to make sure we don't enforce the contraints
                        # immediately.
                        x = last_player_solution.x,
                        u = last_player_solution.u,
                        λ = last_player_solution.λ,
                    ),
                    inner_solver_kwargs...,
                )

            player_weights[player_idx] = last_player_solution.weights
        end

        converged =
            sum(Δu -> Δu^2, last_player_solution.u .- last_ibr_solution.u) <=
            ibr_convergence_tolerance
        last_ibr_solution = last_player_solution

        if converged
            verbose && @info "Converged at ibr iterate: $i_ibr"
            break
        end
    end

    converged || @warn "IBR terminated pre-maturely."

    converged, last_ibr_solution, player_opt_models, player_weights
end

#================================ Inverse Games via KKT constraints ================================#

struct InverseKKTConstraintSolver end

function solve_inverse_game(
    ::InverseKKTConstraintSolver,
    y;
    control_system,
    observation_model,
    player_cost_models,
    init = (),
    solver = Ipopt.Optimizer,
    solver_attributes = (; print_level = 3),
    cmin = 1e-5,
    max_observation_error = nothing,
    init_with_observation = true,
    verbose = false,
)

    T = size(y)[2]
    n_players = length(player_cost_models)
    @unpack n_states, n_controls = control_system

    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # Decision Variables
    player_weights =
        [@variable(opt_model, [keys(cost_model.weights)]) for cost_model in player_cost_models]
    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])
    λ = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])
    # slack variables for unknown initial condition
    x0 = @variable(opt_model, [1:n_states])
    λ0 = @variable(opt_model, [1:n_states, 1:n_players])

    # Initialization
    if init_with_observation
        # TODO: This is not always correct. It will only work if
        # `observation_model.expected_observation` effectively creates an array view into x
        # (extracting components of the variable).
        JuMP.set_start_value.(observation_model.expected_observation(x), y)
    end
    JuMPUtils.init_if_hasproperty!(u, init, :u)
    JuMPUtils.init_if_hasproperty!(λ, init, :λ)

    # # TODO: think about initialization for player weights. Make this a kwarg
    # for weights in player_weights
    #     JuMP.set_start_value.(weights, 1 / length(weights))
    # end

    # constraints
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

    if iszero(observation_model.σ)
        @constraint(opt_model, observation_model.expected_observation(x0) .== y[:, 1])
    end
    @constraint(opt_model, x[:, 1] .== x0)

    for (player_idx, cost_model) in enumerate(player_cost_models)
        weights = player_weights[player_idx]
        @unpack player_inputs = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

        # KKT Nash constraints
        @constraint(
            opt_model,
            dJ.dx[:, 1] - (λ[:, 1, player_idx]' * df.dx[:, :, 1])' + λ0[:, player_idx] .== 0
        )
        @constraint(
            opt_model,
            [t = 2:(T - 1)],
            dJ.dx[:, t] + λ[:, t - 1, player_idx] - (λ[:, t, player_idx]' * df.dx[:, :, t])' .== 0
        )
        @constraint(opt_model, dJ.dx[:, T] + λ[:, T - 1, player_idx] .== 0)

        @constraint(
            opt_model,
            [t = 1:(T - 1)],
            dJ.du[player_inputs, t] - (λ[:, t, player_idx]' * df.du[:, player_inputs, t])' .== 0
        )
        @constraint(opt_model, dJ.du[player_inputs, T] .== 0)
    end

    # regularization
    for weights in player_weights
        @constraint(opt_model, weights .>= cmin)
        @constraint(opt_model, sum(weights) .== 1)
    end

    y_expected = observation_model.expected_observation(x)
    # TODO: dirty hack
    # Only search in a local neighborhood of the demonstration if we have an error-bound on the
    # noise.
    if !isnothing(max_observation_error)
        @constraint(opt_model, (y_expected - y) .^ 2 .<= max_observation_error^2)
    end

    # The inverse objective: match the observed demonstration
    @objective(opt_model, Min, sum(el -> el^2, y_expected .- y))

    time = @elapsed JuMP.optimize!(opt_model)
    verbose && @info time

    solution = merge(
        JuMPUtils.get_values(; x, u, λ),
        (; player_weights = map(w -> CostUtils.namedtuple(JuMP.value.(w)), player_weights)),
    )

    JuMPUtils.isconverged(opt_model), solution, opt_model
end

#========================================== KKT Residual ===========================================#

struct InverseKKTResidualSolver end

function solve_inverse_game(
    ::InverseKKTResidualSolver,
    x,
    u;
    control_system,
    player_cost_models,
    init = (),
    solver = Ipopt.Optimizer,
    solver_attributes = (; print_level = 3),
    cmin = 1e-5,
    verbose = false,
)

    T = size(x)[2]
    n_players = length(player_cost_models)
    @unpack n_states, n_controls = control_system

    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # Decision Variables
    player_weights =
        [@variable(opt_model, [keys(cost_model.weights)]) for cost_model in player_cost_models]
    λ = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])

    # Initialization
    JuMPUtils.init_if_hasproperty!(λ, init, :λ)
    # # TODO: think about initialization for player weights. Make this a kwarg
    # for weights in player_weights
    #     JuMP.set_start_value.(weights, 1 / length(weights))
    # end

    # Compute intermediate results needed for the KKT residual
    # NOTE: In the KKT residual formulation, it is an *unconstrained* optimizatino problem. The only
    # constraints that we add here are the reiguarization constraints ont he palyer weights.
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

    player_residuals = map(enumerate(player_cost_models)) do (player_idx, cost_model)
        weights = player_weights[player_idx]
        @unpack player_inputs = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

        dLdx = let
            # Note: The first of these is never used but 1-based indexing is needed to allow
            # broadcasting.
            dLdx = @variable(opt_model, [1:n_states, 1:T])

            @constraint(
                opt_model,
                [t = 2:(T - 1)],
                dLdx[:, t] .==
                dJ.dx[:, t] + λ[:, t - 1, player_idx] - (λ[:, t, player_idx]' * df.dx[:, :, t])'
            )
            @constraint(opt_model, dLdx[:, T] .== dJ.dx[:, T] + λ[:, T - 1, player_idx])

            dLdx
        end

        dLdu = let
            dLdu = @variable(opt_model, [1:length(player_inputs), 1:T])
            @constraint(
                opt_model,
                [t = 1:(T - 1)],
                dLdu[:, t] .==
                dJ.du[player_inputs, t] - (λ[:, t, player_idx]' * df.du[:, player_inputs, t])'
            )
            @constraint(opt_model, dLdu[:, T] .== dJ.du[player_inputs, T])

            dLdu
        end

        (; dLdx, dLdu)
    end

    # regularization
    for weights in player_weights
        @constraint(opt_model, weights .>= cmin)
        @constraint(opt_model, sum(weights) .== 1)
    end

    @objective(
        opt_model,
        Min,
        sum(sum(el -> el^2, res.dLdx) + sum(el -> el^2, res.dLdu) for res in player_residuals)
    )

    time = @elapsed JuMP.optimize!(opt_model)
    verbose && @info time

    solution = merge(
        JuMPUtils.get_values(; λ),
        (; player_weights = map(w -> CostUtils.namedtuple(JuMP.value.(w)), player_weights)),
    )

    JuMPUtils.isconverged(opt_model), solution, opt_model
end

end
