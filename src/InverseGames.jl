module InverseGames

using JuMP: JuMP
using Ipopt: Ipopt
using ..DynamicsModelInterface: DynamicsModelInterface
using ..JuMPUtils: JuMPUtils
using ..CostUtils: CostUtils
using ..InverseOptimalControl: InverseOptimalControl
using ..InversePreSolve: InversePreSolve
using ..ForwardGame: ForwardGame

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game

#================================ Inverse Games via KKT constraints ================================#

struct InverseKKTConstraintSolver end

function solve_inverse_game(
    ::InverseKKTConstraintSolver,
    y;
    control_system,
    observation_model,
    player_cost_models,
    T_predict = 0,
    init = (),
    solver = Ipopt.Optimizer,
    solver_attributes = (; print_level = 3),
    cmin = 1e-5,
    max_observation_error = nothing,
    prior = nothing,
    prior_weight = 1e-3,
    init_with_observation = true,
    verbose = false,
    pre_solve = true,
    pre_solve_kwargs = (;),
)
    T_predict >= 0 ||
        throw(ArgumentError("The prediction horizon `T_predict` must be non-negative."))
    T = size(y, 2) + T_predict

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

    if pre_solve
        pre_solve_converged, pre_solve_init = InversePreSolve.pre_solve(
            y,
            nothing;
            control_system,
            observation_model,
            T,
            verbose,
            init,
            solver,
            solver_attributes,
            pre_solve_kwargs...,
        )
        @assert pre_solve_converged
        # TODO: think about how to set an initial guess for the tail end. Maybe Just constant
        # velocity rollout?
        JuMP.set_start_value.(@view(x[CartesianIndices(pre_solve_init.x)]), pre_solve_init.x)
    elseif init_with_observation
        # Note: This is not always correct. It will only work if
        # `observation_model.expected_observation` effectively creates an array view into x
        # (extracting components of the variable).
        JuMP.set_start_value.(observation_model.expected_observation(x), y)

        # TODO maybe also warm-start the state and input estimates
        JuMPUtils.init_if_hasproperty!(x, init, :x)
        JuMPUtils.init_if_hasproperty!(u, init, :u)
        JuMPUtils.init_if_hasproperty!(λ, init, :λ)
    end

    if hasproperty(init, :player_weights) && !isnothing(init.player_weights)
        for (ii, weights) in pairs(init.player_weights)
            for k in keys(weights)
                JuMP.set_start_value(player_weights[ii][k], weights[k])
            end
        end
    end

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

    y_expected = observation_model.expected_observation(x)[:, 1:size(y, 2)]
    # Sometimes useful for debugging: Only search in a local neighborhood of the demonstration if we
    # have an error-bound on the noise.
    if !isnothing(max_observation_error)
        @constraint(opt_model, (y_expected - y) .^ 2 .<= max_observation_error^2)
    end

    # The inverse objective: match the observed demonstration
    prior_penalty = if !isnothing(prior)
        weight_prior = if haskey(prior, :player_weights)
            sum(el -> el^2, y_expected .- y) +
            sum(zip(player_weights, prior.player_weights)) do (w, w_prior)
                sum(w[k] - w_prior[k] for k in keys(w_prior))
            end
        else
            0
        end

        state_prior = if haskey(prior, :x)
            sum(el -> el^2, x[:, 1:size(prior.x, 2)] .- prior.x)
        else
            0
        end

        input_prior = if haskey(prior, :x)
            sum(el -> el^2, u[:, 1:size(prior.u, 2)] .- prior.u)
        else
            0
        end

        # TODO implement proper weighting of prior
        verbose && @warn "Note: The prior is only weighted heuristically for now."
        (weight_prior + state_prior + input_prior) * prior_weight
    else
        0
    end
    @objective(opt_model, Min, sum(el -> el^2, y_expected .- y))

    time = @elapsed JuMP.optimize!(opt_model)
    verbose && @info time

    solution = merge(
        JuMPUtils.get_values(; x, u, λ),
        (; player_weights = map(w -> CostUtils.namedtuple(JuMP.value.(w)), player_weights)),
    )

    (JuMPUtils.isconverged(opt_model), solution, opt_model)
end

#========================================== KKT Residual ===========================================#
#
struct AugmentedInverseKKTResidualSolver end

# TODO: partially observed dispatch
# Maybe this should be another solver? Like `FilteringInverseKKTResidualSolver`
function solve_inverse_game(
    inverse_solver::AugmentedInverseKKTResidualSolver,
    y;
    control_system,
    observation_model,
    player_cost_models,
    solver = Ipopt.Optimizer,
    solver_attributes = (; print_level = 3),
    prediction_solver = ForwardGame.IBRGameSolver(),
    verbose = false,
    pre_solve_kwargs = (;),
    # TODO: implement these features...
    T_predict = 0,
    prior = nothing,
    max_observation_error = nothing,
    solver_args...,
)
    T_predict >= 0 ||
        throw(ArgumentError("The prediction horizon `T_predict` must be non-negative."))
    T_obs = size(y, 2)
    T = T_obs + T_predict

    converged = false

    pre_solve_converged, pre_solve_solution = InversePreSolve.pre_solve(
        y,
        nothing;
        control_system,
        observation_model,
        solver_attributes,
        verbose,
        pre_solve_kwargs...,
        T,
    )
    @assert pre_solve_converged
    converged = pre_solve_converged

    # Filtered sequence is truncated to the original length to give all methods the same
    # number of data-points for inference.
    # TODO: Think about how to cleanly handle the "extra observation" case here. Otherwise, just use
    # the `InverseKKTResidualSolver` dispatch for the old experiments.
    smoothed_observation =
        (; x = pre_solve_solution.x[:, 1:T_obs], u = pre_solve_solution.u[:, 1:T_obs])

    inverse_converged, estimate, opt_model = solve_inverse_game(
        InverseKKTResidualSolver(),
        smoothed_observation.x,
        smoothed_observation.u;
        control_system,
        player_cost_models,
        solver,
        solver_attributes,
        verbose,
        solver_args...,
    )
    @assert inverse_converged
    converged = converged && inverse_converged

    trajectory = smoothed_observation
    if T_predict > 0
        # TODO: assemble the `estimated_player_cost_models`.
        prediction_init =
            (; x = pre_solve_solution.x[:, T_obs:end], u = pre_solve_solution.u[:, T_obs:end])
        x0_predict = prediction_init.x[:, begin]

        estimated_player_cost_models =
            map(player_cost_models, estimate.player_weights) do cost_model_gt, weights
                merge(cost_model_gt, (; weights))
            end
        prediction_converged, prediction = ForwardGame.solve_game(
            prediction_solver,
            control_system,
            estimated_player_cost_models,
            x0_predict,
            T_predict + 1;
            solver,
            solver_attributes,
            init = prediction_init,
            verbose,
        )

        converged = converged && prediction_converged
        trajectory = (;
            x = [smoothed_observation.x prediction.x[:, 2:end]],
            u = [smoothed_observation.u prediction.u[:, 2:end]],
        )

        trajectory
    end

    converged, (; trajectory..., estimate...), opt_model
end

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
    T = size(x, 2)
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
        sum(
            sum(el -> el^2, res.dLdx) + sum(el -> el^2, res.dLdu[:, 1:(end - 1)]) for
            res in player_residuals
        )
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
