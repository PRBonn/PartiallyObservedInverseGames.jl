module ForwardGame

import JuMP
import Ipopt

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack
using ..ForwardOptimalControl: solve_optimal_control
using ..SolverUtils

export solve_ol_nash_ibr

#================================ Iterated Best Open-Loop Response =================================#

# TODO: We could allow to pass an inner solver
# TODO: make solution technique (e.g. KKT vs IBR) a dispatch argument
function solve_ol_nash_ibr(
    control_system,
    cost_models,
    x0,
    T;
    inner_solver_kwargs = (),
    max_ibr_rounds = 10,
    ibr_convergence_tolerance = 0.01,
)
    @unpack n_states, n_controls = control_system
    player_indices = eachindex(cost_models)
    last_ibr_solution = (; x = zeros(n_states, T), u = zeros(n_controls, T))
    last_player_solution = last_ibr_solution
    converged = false
    player_opt_models = Any[nothing, nothing]

    for i_ibr in 1:max_ibr_rounds
        for (player_idx, player_cost_model) in enumerate(cost_models)
            last_player_solution, player_opt_models[player_idx] = solve_optimal_control(
                control_system,
                player_cost_model,
                x0,
                T;
                fix_inputs = filter(!=(player_idx), player_indices),
                init = last_player_solution,
                inner_solver_kwargs...,
            )
        end

        converged =
            sum(x -> x^2, last_player_solution.x - last_ibr_solution.x) <= ibr_convergence_tolerance
        last_ibr_solution = last_player_solution

        if converged
            @info "Converged at ibr iterate: $i_ibr"
            break
        end
    end

    last_ibr_solution, converged, player_opt_models
end

#================================= Open-Loop KKT Nash Constraints ==================================#

# TODO: Share containers between players to make implementation more generic (agnostic to different
# numbers of players) (u's and λ's)
# TODO handle missing "init" keys more gracefully (also in other solvers)
function solve_ol_nash_kkt(
    control_system,
    cost_model,
    x0,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
    init = (; λ1 = nothing, λ2 = nothing, x = nothing, u = nothing),
)
    @unpack n_states = control_system
    model = JuMP.Model(solver)
    SolverUtils.set_solver_attributes!(model; silent, solver_attributes...)

    # Decision Variables
    # TODO: fix the variable access here.
    x = @variable(model, x[1:n_states, 1:T])
    u1 = @variable(model, u1[1:1, 1:T])
    u2 = @variable(model, u2[1:1, 1:T])
    # TODO: think about where/if we have to share lagrange multipliers
    λ1 = @variable(model, λ1[1:n_states, 1:T-1])
    λ2 = @variable(model, λ2[1:n_states, 1:T-1])
    u = [u1; u2]

    # Initialization
    isnothing(init.λ1) || JuMP.set_start_value.(λ1, init.λ1)
    isnothing(init.λ2) || JuMP.set_start_value.(λ2, init.λ2)
    isnothing(init.x) || JuMP.set_start_value.(x, init.x)
    isnothing(init.u) || JuMP.set_start_value.(u, init.u)

    # constraints
    @constraint(model, x[:, 1] .== x0)
    control_system.add_dynamics_constraints!(model, x, u)
    df = control_system.add_dynamics_jacobians!(model, x, u)
    dJ1 = cost_model.objective_gradients_p1(x, u1; cost_model.weights)
    dJ2 = cost_model.objective_gradients_p2(x, u2; cost_model.weights)
    # TODO: figure out whether/which multipliers need to be shared
    # P1 KKT
    dL1dx = [dJ1.dx[:, t] + λ1[:, t - 1] - df.dx[:, :, t]' * λ1[:, t] for t in 2:T-1]
    dL1du1 = [dJ1.du1[:, t] - (df.du[:, 1:1, t]' * λ1[:, t]) for t in 1:T-1]
    @constraint(model, [t = eachindex(dL1dx)], dL1dx[t] .== 0)
    @constraint(model, [t = eachindex(dL1du1)], dL1du1[t] .== 0)
    # P2 KKT
    dL2dx = [dJ2.dx[:, t] + λ2[:, t - 1] - df.dx[:, :, t]' * λ2[:, t] for t in 2:T-1]
    dL2du2 = [dJ2.du2[:, t] - (df.du[:, 2:2, t]' * λ2[:, t]) for t in 1:T-1]
    @constraint(model, [t = eachindex(dL2dx)], dL2dx[t] .== 0)
    @constraint(model, [t = eachindex(dL2du2)], dL2du2[t] .== 0)

    @time JuMP.optimize!(model)
    SolverUtils.get_model_values(model, :x, :u1, :u2, :λ1, :λ2), model
end

function solve_ol_nash_kkt_cloning(
    control_system,
    cost_model,
    x0,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
    init = (; λ1 = nothing, λ2 = nothing, x = nothing, u = nothing),
)
    @unpack n_states = control_system
    model = JuMP.Model(solver)
    SolverUtils.set_solver_attributes!(model; silent, solver_attributes...)

    # P1's copy of the decision variables
    x1 = @variable(model, x1[1:n_states, 1:T])
    u1 = @variable(model, u1[1:1, 1:T])
    λ1 = @variable(model, λ1[1:n_states, 1:T-1])
    isnothing(init.x) || JuMP.set_start_value.(x1, init.x)
    isnothing(init.u) || JuMP.set_start_value.(u1, init.u[1, :])
    isnothing(init.λ1) || JuMP.set_start_value.(λ1, init.λ1)

    # P2
    x2 = @variable(model, x2[1:n_states, 1:T])
    u2 = @variable(model, u2[1:1, 1:T])
    λ2 = @variable(model, λ2[1:n_states, 1:T-1])
    isnothing(init.x) || JuMP.set_start_value.(x2, init.x)
    isnothing(init.u) || JuMP.set_start_value.(u2, init.u[2, :])
    isnothing(init.λ2) || JuMP.set_start_value.(λ2, init.λ2)

    # shared quantities
    x = @variable(model, x[1:n_states, 1:T])
    u = [u1; u2]
    @constraint(model, x[:, 1] .== x0)
    control_system.add_dynamics_constraints!(model, x, u)
    df = control_system.add_dynamics_jacobians!(model, x, u)

    # coupling constriants
    # TODO: maybe also clone full `u`
    @constraint(model, x .== x1)
    @constraint(model, x .== x2)

    # P1 constraints
    dJ1 = cost_model.objective_gradients_p1(x1, u1; cost_model.weights)
    dL1dx = [dJ1.dx[:, t] + λ1[:, t - 1] - df.dx[:, :, t]' * λ1[:, t] for t in 2:T-1]
    dL1du1 = [dJ1.du1[:, t] - (df.du[:, 1:1, t]' * λ1[:, t]) for t in 1:T-1]
    @constraint(model, [t = eachindex(dL1dx)], dL1dx[t] .== 0)
    @constraint(model, [t = eachindex(dL1du1)], dL1du1[t] .== 0)

    # P2 constraints
    dJ2 = cost_model.objective_gradients_p2(x2, u2; cost_model.weights)
    dL2dx = [dJ2.dx[:, t] + λ2[:, t - 1] - df.dx[:, :, t]' * λ2[:, t] for t in 2:T-1]
    dL2du2 = [dJ2.du2[:, t] - (df.du[:, 1:1, t]' * λ2[:, t]) for t in 1:T-1]
    @constraint(model, [t = eachindex(dL2dx)], dL2dx[t] .== 0)
    @constraint(model, [t = eachindex(dL2du2)], dL2du2[t] .== 0)

    @time JuMP.optimize!(model)
    SolverUtils.get_model_values(model, :x, :u1, :u2, :λ1, :λ2), model
end

end
