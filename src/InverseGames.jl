module InverseGames

import JuMP
import Ipopt
import ..SolverUtils

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export solve_inverse_game

# TODO: allow for partial and noisy state observations
function solve_inverse_game(
    x̂,
    û = nothing;
    control_system,
    player_cost_models,
    λ_init = nothing,
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

    # Initialization
    JuMP.set_start_value.(x, x̂)
    !isnothing(û) && JuMP.set_start_value.(u, û)
    !isnothing(λ_init) && JuMP.set_start_value.(λ, λ_init)
    for weights in player_weights
        JuMP.set_start_value.(weights, 1 / length(weights))
    end

    # constraints
    control_system.add_dynamics_constraints!(model, x, u)
    df = control_system.add_dynamics_jacobians!(model, x, u)

    for (player_idx, cost_model) in enumerate(player_cost_models)
        weights = player_weights[player_idx]
        @unpack player_inputs = cost_model
        dJ = cost_model.objective_gradients(x, u; weights)

        # KKT Nash constraints
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

        # TODO: Think about the correct regularization here
        @constraint(model, sum(weights) .== 1)
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
