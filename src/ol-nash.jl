import JuMP
import Ipopt
import Zygote

using Test: @test, @testset
using JuMP: @variable, @constraint, @objective
using UnPack: @unpack
using SparseArrays: spzeros

import Plots, ElectronDisplay

include("utils.jl")
include("unicycle.jl")

#======================================== Global parameters ========================================#

function objective_p1(x, u1; weights)
    weights[:state_velocity_p1] * sum((x[3, :] .- 0.1) .^ 2) +
    weights[:control_Δv_p1] * sum(u1 .^ 2)
end

function objective_gradients_p1(x, u1; weights)
    T = size(x, 2)
    dJ1dx = 2 * weights[:state_velocity_p1] * [zeros(2, T); x[3:3, :] .- 0.1; zeros(1, T)]
    dJ1du1 = 2 * weights[:control_Δv_p1] * u1
    (; dx = dJ1dx, du1 = dJ1du1)
end

function objective_p2(x, u2; weights)
    weights[:state_goal_p2] * sum(x[1:2, :] .^ 2) + weights[:control_Δθ_p2] * sum(u2 .^ 2)
end

function objective_gradients_p2(x, u2; weights)
    T = size(x, 2)
    dJ2dx = 2 * [x[1:2, :]; zeros(2, T)] * weights[:state_goal_p2]
    dJ2du2 = 2 * u2 * weights[:control_Δθ_p2]
    (; dx = dJ2dx, du2 = dJ2du2)
end

control_system = (
    add_dynamics_constraints! = add_unicycle_dynamics_constraints!,
    add_dynamics_jacobians! = add_unicycle_dynamics_jacobians!,
    n_states = 4,
    n_controls = 2,
)

x0 = [-1, 1, 0.1, 0]
T = 100
cost_model = (;
    weights = (; state_goal_p2 = 1, state_velocity_p1 = 10, control_Δv_p1 = 10, control_Δθ_p2 = 1),
    # TODO: remove. Dummy objective to solve fully-collaborative version of the game.
    add_objective! = function (model, x, u; weights)
        @objective(model, Min, objective_p1(x, u; weights) + objective_p2(x, u; weights))
    end,
    objective_p1,
    objective_gradients_p1,
    objective_p2,
    objective_gradients_p2,
)

#====================================== forward optimal control ====================================#

function solve_ol_nash_kkt(
    control_system,
    cost_model,
    x0,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
)

    @unpack n_states = control_system
    model = JuMP.Model(solver)
    set_solver_attributes!(model; silent, solver_attributes...)

    # TODO: fix the variable access here.
    x = @variable(model, x[1:n_states, 1:T])
    u1 = @variable(model, u1[1:1, 1:T])
    u2 = @variable(model, u2[1:1, 1:T])
    # TODO: think about where/if we have to share lagrange multipliers
    λ1 = @variable(model, λ1[1:n_states, 1:T])
    λ2 = @variable(model, λ2[1:n_states, 1:T])

    # TODO: fix ugly hack. Hard-code a dynamically feasible initial trajectory
    # x_init = reduce(1:(T - 1); init = x0) do x, t
    #     [x x[:, end] + [x0[3], 0, 0, 0]]
    # end
    JuMP.set_start_value.(λ1, 0)
    JuMP.set_start_value.(x, x_init)
    JuMP.set_start_value.(u1, u_init[1, :])
    JuMP.set_start_value.(u2, u_init[2, :])

    u = [u1; u2]

    # constraints
    # Initial condition
    @constraint(model, x[:, 1] .== x0)
    #TODO: continue here
    # Joint state feasibility
    control_system.add_dynamics_constraints!(model, x, u)
    df = control_system.add_dynamics_jacobians!(model, x, u)
    dJ1 = cost_model.objective_gradients_p1(x, u1; cost_model.weights)
    dJ2 = cost_model.objective_gradients_p2(x, u2; cost_model.weights)

    # TODO: figure out whether/which multipliers need to be shared
    # P1 KKT
    dL1dx = [dJ1.dx[:, t] + λ1[:, t - 1] - df.dx[:, :, t]' * λ1[:, t] for t in 2:T]
    dL1du1 = [dJ1.du1[:, t] - (df.du[:, 1:1, t]' * λ1[:, t]) for t in 2:T]
    @constraint(model, [t = eachindex(dL1dx)], dL1dx[t] .== 0)
    @constraint(model, [t = eachindex(dL1du1)], dL1du1[t] .== 0)
    # P2 KKT
    dL2dx = [dJ2.dx[:, t] + λ2[:, t - 1] - df.dx[:, :, t]' * λ2[:, t] for t in 2:T]
    dL2du2 = [dJ2.du2[:, t] - (df.du[:, 2:2, t]' * λ2[:, t]) for t in 2:T]
    @constraint(model, [t = eachindex(dL2dx)], dL2dx[t] .== 0)
    @constraint(model, [t = eachindex(dL2du2)], dL2du2[t] .== 0)

    @time JuMP.optimize!(model)
    get_model_values(model, :x, :u1, :u2, :λ1, :λ2), model
end

@testset "Gradient check" begin
    x = rand(4, 100)
    u = rand(2, 100)

    J1 = objective_p1(x, u[1, :]; cost_model.weights)
    dJ1dx, dJ1du1 = Zygote.gradient((x, u1) -> objective_p1(x, u1; cost_model.weights), x, u[1, :])
    dJ1 = objective_gradients_p1(x, u[1, :]; cost_model.weights)
    @test dJ1dx == dJ1.dx
    @test dJ1du1 == dJ1.du1

    J2 = objective_p2(x, u[2, :]; cost_model.weights)
    dJ2dx, dJ2du2 = Zygote.gradient((x, u2) -> objective_p2(x, u2; cost_model.weights), x, u[2, :])
    dJ2 = objective_gradients_p2(x, u[2, :]; cost_model.weights)
    @test dJ2dx == dJ2.dx
    @test dJ2du2 == dJ2.du2
end

# TODO: debug... does not converge right now.
forward_solution, forward_model = solve_ol_nash(control_system, cost_model, x0, T)

end

# TODO: debug... does not converge right now.
forward_solution, forward_model = solve_ol_nash_kkt(control_system, cost_model, x0, T)

# TODO: check forward optimal solution with single-player implementation (this could also be solved
# as an optimal problem)
#   - If that works, see if we can warmstart the solver
