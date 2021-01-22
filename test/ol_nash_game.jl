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
        (;
            add_objective! = function (model, x, u; weights)
                @objective(model, Min, cost_model.objective_p1(x, u[1, :]; weights))
            end,
        ),
    ),
    # cost_model P2
    merge(
        cost_model,
        (;
            add_objective! = function (model, x, u; weights)
                @objective(model, Min, cost_model.objective_p2(x, u[2, :]; weights))
            end,
        ),
    ),
)

#=============================================== Tests =============================================#

@testset "Gradient check" begin
    x = rand(4, 100)
    u = rand(2, 100)

    dJ1dx_ad, dJ1du1_ad =
        Zygote.gradient((x, u1) -> objective_p1(x, u1; cost_model.weights), x, u[1, :])
    dJ1 = objective_gradients_p1(x, u[1, :]; cost_model.weights)
    @test dJ1dx_ad == dJ1.dx
    @test dJ1du1_ad == dJ1.du1

    dJ2dx_ad, dJ2du2_ad =
        Zygote.gradient((x, u2) -> objective_p2(x, u2; cost_model.weights), x, u[2, :])
    dJ2 = objective_gradients_p2(x, u[2, :]; cost_model.weights)
    @test dJ2dx_ad == dJ2.dx
    @test dJ2du2_ad == dJ2.du2
end

@testset "Iterated Best Open-Loop Response" begin

    global ibr_nash, ibr_converged = solve_ol_nash_ibr(
        control_system,
        player_cost_models,
        x0,
        T;
        inner_solver_kwargs = (; silent = true),
    )

    @test ibr_converged
end

# precompute lagrange multipliers
begin

    df = let x = ibr_nash.x
        As = [
            [
                1 0 cos(x[4, t]) -x[3, t]*sin(x[4, t])
                0 1 sin(x[4, t]) +x[3, t]*cos(x[4, t])
                0 0 1 0
                0 0 0 1
            ] for t in 1:T
        ]

        Bs = [[
            0 0
            0 0
            1 0
            0 1
        ] for t in 1:T]

        (;
            dx = reduce((A, x) -> cat(A, x; dims = 3), As),
            du = reduce((A, x) -> cat(A, x; dims = 3), Bs),
        )
    end

    λ1 = zeros(control_system.n_states, T)
    @testset "λ1" begin
        dJ1 = cost_model.objective_gradients_p1(ibr_nash.x, ibr_nash.u[1:1, :]; cost_model.weights)

        dL1dx = [dJ1.dx[:, t] + λ1[:, t - 1] - df.dx[:, :, t]' * λ1[:, t] for t in 2:T]
        dL1du1 = [dJ1.du1[:, t] - (df.du[:, 1:1, t]' * λ1[:, t]) for t in 2:T]

        @test all(all(isapprox.(x, 0; atol = 1e-10)) for x in dL1dx)
        @test all(all(isapprox.(x, 0; atol = 1e-10)) for x in dL1du1)
    end

    λ2 = let
        model = JuMP.Model(Ipopt.Optimizer)
        λ2 = @variable(model, [1:(control_system.n_states), 1:T])

        # TODO: remove
        # λ[t]' * (x[t+1] - f(x[t], u[t]))

        dJ2 = cost_model.objective_gradients_p2(ibr_nash.x, ibr_nash.u[2:2, :]; cost_model.weights)
        dL2dx = [dJ2.dx[:, t] + λ2[:, t - 1] - df.dx[:, :, t]' * λ2[:, t] for t in 2:T]
        dL2du2 = [dJ2.du2[:, t] - (df.du[:, 2:2, t]' * λ2[:, t]) for t in 1:T]

        @constraint(model, [t = eachindex(dL2dx)], dL2dx[t] .== 0)
        # @constraint(model, [t = eachindex(dL2du2)], dL2du2[t] .== 0)

        JuMP.optimize!(model)
        JuMP.value.(λ2)
    end

    @testset "λ2" begin
        dJ2 = cost_model.objective_gradients_p2(ibr_nash.x, ibr_nash.u[2:2, :]; cost_model.weights)
        dL2dx = [dJ2.dx[:, t] + λ2[:, t - 1] - df.dx[:, :, t]' * λ2[:, t] for t in 2:T]
        dL2du2 = [dJ2.du2[:, t] - (df.du[:, 2:2, t]' * λ2[:, t]) for t in 1:T]

        @test all(all(isapprox.(x, 0; atol = 1e-10)) for x in dL2dx)
        @test all(all(isapprox.(x, 0; atol = 1e-10)) for x in dL2du2)
    end
end

# TODO: debug... does not converge right now.
# - warm start the solver with the IBR solution
#   - [done] only initialize "primal" variables (x, u)
#   - also initialize Lagrange multipliers `λ`
#       - [done] λ1 should be zero,
#       - not sure how to compute `λ2` (@DFK)
#   - [done] reduce the solver tolerance
#   - check multipliers from IBR
#       - same as manually computed up to a sign
#       - 0 for P1
#       - non-zero for P2
#   - try another simpler problem
#       - with decoupled dynamics for P1 and P2
#       - try a fully cooperative version of the problem
kkt_nash, kkt_model = solve_ol_nash_kkt(
    control_system,
    cost_model,
    x0,
    T;
    init = merge(ibr_nash, (; λ1 = 0, λ2 = 1)),
    # solver_attributes = (; constr_viol_tol = 1e-10, tol = 1e-10)
)

# visualize_unicycle_trajectory(kkt_nash.x)
