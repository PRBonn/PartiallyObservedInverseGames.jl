module InverseGames

import JuMP
import Ipopt
import ..SolverUtils
import ..InverseOptimalControl

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export solve_inverse_game

struct InverseIBRSolver end

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
)
    @unpack n_controls = control_system
    n_players = length(player_cost_models)

    last_ibr_solution = (; x = x̂, u = u_init)
    last_player_solution = last_ibr_solution
    player_opt_models = resize!(JuMP.Model[], n_players)
    # TODO: dirty hack
    player_weights = Any[nothing for _ in 1:n_players]
    converged = false

    for i_ibr in 1:max_ibr_rounds
        for (player_idx, player_cost_model) in enumerate(player_cost_models)
            cost_model = player_cost_models[player_idx]
            weights = player_weights[player_idx]

            last_player_solution, player_opt_models[player_idx] =
                InverseOptimalControl.solve_inverse_optimal_control(
                    x̂;
                    control_system,
                    cost_model,
                    observation_model,
                    fixed_inputs = filter(i -> i ∉ player_cost_model.player_inputs, 1:n_controls),
                    init = merge(last_player_solution, (; weights, x = x̂)),
                    inner_solver_kwargs...,
                )

            player_weights[player_idx] = last_player_solution.weights
        end

        converged =
            i_ibr > 1 &&
            sum(u -> u^2, last_player_solution.u .- last_ibr_solution.u) <=
            ibr_convergence_tolerance
        last_ibr_solution = last_player_solution

        if converged
            @info "Converged at ibr iterate: $i_ibr"
            break
        end
    end

    converged || @warn "IBR terminated pre-maturely."

    converged, last_ibr_solution, player_opt_models, player_weights
end

#================================ Inverse Games via KKT constraints ================================#

struct InverseKKTSolver end

# TODO: allow for partial and noisy state observations
function solve_inverse_game(
    ::InverseKKTSolver,
    x̂,
    û = nothing;
    control_system,
    player_cost_models,
    init = (),
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
    cmin = 1e-5,
    max_trajectory_error = nothing,
)

    T = size(x̂)[2]
    n_players = length(player_cost_models)
    @unpack n_states, n_controls = control_system

    model = JuMP.Model(solver)
    SolverUtils.set_solver_attributes!(model; silent, solver_attributes...)

    # Decision Variables
    # TODO: continue here: add and intialize weights (key as union over player cost weight keys)
    player_weights =
        [@variable(model, [keys(cost_model.weights)]) for cost_model in player_cost_models]
    x = @variable(model, [1:n_states, 1:T])
    u = @variable(model, [1:n_controls, 1:T])
    λ = @variable(model, [1:n_states, 1:(T - 1), 1:n_players])

    x0 = @variable(model, [1:n_states])
    λ0 = @variable(model, [1:n_states, 1:n_players])

    # Initialization
    JuMP.set_start_value.(x, x̂)
    SolverUtils.init_if_hasproperty!(u, init, :u)
    SolverUtils.init_if_hasproperty!(λ, init, :λ)

    # TODO: think about initialization for player weights
    # for weights in player_weights
    #     JuMP.set_start_value.(weights, 1 / length(weights))
    # end

    # constraints
    control_system.add_dynamics_constraints!(model, x, u)
    df = control_system.add_dynamics_jacobians!(model, x, u)

    @constraint(model, x[:, 1] .== x0)

    for (player_idx, cost_model) in enumerate(player_cost_models)
        weights = player_weights[player_idx]
        @unpack player_inputs = cost_model
        dJ = cost_model.objective_gradients(x, u; weights)

        # KKT Nash constraints
        @constraint(
            model,
            dJ.dx[:, 1] - (λ[:, 1, player_idx]' * df.dx[:, :, 1])' + λ0[:, player_idx] .== 0
        )
        @constraint(
            model,
            [t = 2:(T - 1)],
            dJ.dx[:, t] + λ[:, t - 1, player_idx] - (λ[:, t, player_idx]' * df.dx[:, :, t])' .== 0
        )
        @constraint(model, dJ.dx[:, T] + λ[:, T - 1, player_idx] .== 0)

        @constraint(
            model,
            [t = 1:(T - 1)],
            dJ.du[player_inputs, t] - (λ[:, t, player_idx]' * df.du[:, player_inputs, t])' .== 0
        )
        @constraint(model, dJ.du[player_inputs, T] .== 0)

        # regularization
        @constraint(model, weights .>= cmin)
        # # TODO: Think about the correct regularization here
        @constraint(model, sum(w -> w^2, weights) .== 1)
    end

    # TODO: Think about the correct regularization here
    # @constraint(model, sum(sum(weights) for weights in player_weights) .== 1)

    # The inverse objective: match the observed demonstration
    @objective(model, Min, sum((x .- x̂) .^ 2))

    @time JuMP.optimize!(model)
    merge(
        SolverUtils.get_values(; x, u, λ),
        (; player_weights = map(w -> JuMP.value.(w), player_weights)),
    ),
    model
end

end
