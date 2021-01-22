import JuMP
import Ipopt
import Zygote

using JuMP: @variable, @constraint, @objective
using JuMPOptimalControl.ForwardGame: solve_ol_nash_ibr, solve_ol_nash_kkt
using SparseArrays: spzeros
using Test: @test, @testset
using UnPack: @unpack

import Plots, ElectronDisplay

include("Unicycle.jl")
using .Unicycle:
    add_unicycle_dynamics_constraints!,
    add_unicycle_dynamics_jacobians!,
    visualize_unicycle_trajectory

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
    objective_p1,
    objective_gradients_p1,
    objective_p2,
    objective_gradients_p2,
)

player_cost_models = (
    # cost_model P1
    merge(
        cost_model,
        (; add_objective! = function (model, x, u; weights)
            @objective(model, Min, objective_p1(x, u[1, :]; weights))
        end),
    ),
    # cost_model P2
    merge(
        cost_model,
        (; add_objective! = function (model, x, u; weights)
            @objective(model, Min, objective_p2(x, u[2, :]; weights))
        end),
    ),
)

#====================================== forward optimal control ====================================#

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

ibr_nash, ibr_converged = solve_ol_nash_ibr(
    control_system,
    player_cost_models,
    x0,
    T;
    inner_solver_kwargs = (; silent = true),
)

@test ibr_converged

# TODO: fix ugly hack. Hard-code a dynamically feasible initial trajectory
# x_init = reduce(1:(T - 1); init = x0) do x, t
#     [x x[:, end] + [x0[3], 0, 0, 0]]
# end

# TODO: debug... does not converge right now.
kkt_nash, kkt_model = solve_ol_nash_kkt(control_system, cost_model, x0, T)

# TODO: check forward optimal solution with single-player implementation (this could also be solved
# as an optimal problem)
#   - If that works, see if we can warmstart the solver
