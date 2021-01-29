import JuMP
import Ipopt
import Zygote

using JuMP: @variable, @constraint, @objective
using JuMPOptimalControl.ForwardGame: IBRGameSolver, KKTGameSolver, solve_game
using JuMPOptimalControl.InverseGames: InverseIBRSolver, InverseKKTSolver, solve_inverse_game
using JuMPOptimalControl.DynamicsModelInterface: visualize_trajectory
using SparseArrays: spzeros
using Test: @test, @testset, @test_broken
using UnPack: @unpack

import Plots

unique!(push!(LOAD_PATH, @__DIR__))
import TestUtils
import TestDynamics

#======================================== Global parameters ========================================#

function objective_p1(x, u; weights)
    weights[:state_velocity_p1] * sum((x[3, :] .- 0.1) .^ 2) +
    weights[:control_Δv_p1] * sum(u[1, :] .^ 2)
end

function objective_gradients_p1(x, u; weights)
    T = size(x, 2)
    dJdx = 2 * weights[:state_velocity_p1] * [zeros(2, T); x[3:3, :] .- 0.1; zeros(1, T)]
    dJdu = 2 * weights[:control_Δv_p1] * [u[1:1, :]; zeros(1, T)]
    (; dx = dJdx, du = dJdu)
end

function objective_p2(x, u2; weights)
    weights[:state_goal_p2] * sum(x[1:2, :] .^ 2) + weights[:control_Δθ_p2] * sum(u2 .^ 2)
end

function objective_gradients_p2(x, u; weights)
    T = size(x, 2)
    dJdx = 2 * weights[:state_goal_p2] * [x[1:2, :]; zeros(2, T)]
    dJdu = 2 * weights[:control_Δθ_p2] * [zeros(1, T); u[2:2, :]]
    (; dx = dJdx, du = dJdu)
end

control_system = TestDynamics.Unicycle()
x0 = [-1, 1, 0.0, 0]
T = 100

player_cost_models = (
    (;
        player_inputs = [1],
        weights = (; state_velocity_p1 = 10, control_Δv_p1 = 100),
        objective = objective_p1,
        objective_gradients = objective_gradients_p1,
        add_objective! = function (opt_model, args...; kwargs...)
            @objective(opt_model, Min, objective_p1(args...; kwargs...))
        end,
        add_objective_gradients! = function (opt_model, args...; kwargs...)
            objective_gradients_p1(args...; kwargs...)
        end,
    ),
    (;
        player_inputs = [2],
        weights = (; state_goal_p2 = 0.1, control_Δθ_p2 = 10),
        objective = objective_p2,
        objective_gradients = objective_gradients_p2,
        add_objective! = function (opt_model, args...; kwargs...)
            @objective(opt_model, Min, objective_p2(args...; kwargs...))
        end,
        add_objective_gradients! = function (opt_model, args...; kwargs...)
            objective_gradients_p2(args...; kwargs...)
        end,
    ),
)

#=============================================== Tests =============================================#

@testset "Gradient check" begin
    x = rand(4, 100)
    u = rand(2, 100)

    for (player_idx, cost_model) in enumerate(player_cost_models)
        dJdx_ad, dJdu_ad =
            Zygote.gradient((x, u) -> cost_model.objective(x, u; cost_model.weights), x, u)
        dJ = cost_model.objective_gradients(x, u; cost_model.weights)
        @test dJdx_ad == dJ.dx
        @test dJdu_ad[player_idx, :] == dJ.du[player_idx, :]
    end
end

function test_unicycle_multipliers(λ, x, u; player_cost_models)
    df = let
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

    for (player_idx, cost_model) in enumerate(player_cost_models)
        @testset "λ$player_idx" begin
            dJ = cost_model.objective_gradients(x, u; cost_model.weights)

            dLdx = [
                dJ.dx[:, t] + λ[:, t - 1, player_idx] - df.dx[:, :, t]' * λ[:, t, player_idx]
                for t in 2:(T - 1)
            ]
            dLdu = [
                dJ.du[cost_model.player_inputs, t] -
                (df.du[:, cost_model.player_inputs, t]' * λ[:, t, player_idx])
                for t in 1:(T - 1)
            ]

            @test all(all(isapprox.(x, 0; atol = 1e-6)) for x in dLdx)
            @test all(all(isapprox.(x, 0; atol = 1e-6)) for x in dLdu)
        end
    end
end

@testset "Forward IBR" begin
    global ibr_converged, ibr_nash, ibr_models = solve_game(
        IBRGameSolver(),
        control_system,
        player_cost_models,
        x0,
        T;
        inner_solver_kwargs = (; silent = true),
    )
    @test ibr_converged

    # extract constraint multipliers
    global λ_ibr = mapreduce((a, b) -> cat(a, b; dims = 3), ibr_models) do opt_model
        mapreduce(hcat, opt_model[:dynamics]) do c
            # Sign flipped due to internal convention of JuMP
            -JuMP.dual.(c)
        end
    end

    test_unicycle_multipliers(λ_ibr, ibr_nash.x, ibr_nash.u; player_cost_models)
end

@testset "Forward KKT Nash" begin
    global kkt_nash, kkt_model = solve_game(
        KKTGameSolver(),
        control_system,
        player_cost_models,
        x0,
        T;
        solver = Ipopt.Optimizer,
    )

    visualize_trajectory(control_system, kkt_nash.x)

    test_unicycle_multipliers(kkt_nash.λ, kkt_nash.x, kkt_nash.u; player_cost_models)
end

observation_model = (; σ = 0, expected_observation = identity)
# TODO: robustify
@testset "Inverse KKT Nash" begin
    global inverse_kkt_solution, inverse_kkt_model = solve_inverse_game(
        InverseKKTSolver(),
        ibr_nash.x;
        init = (; ibr_nash.u),
        control_system,
        player_cost_models,
    )

    for (cost_model, weights) in zip(player_cost_models, inverse_kkt_solution.player_weights)
        TestUtils.test_inverse_solution(weights, cost_model.weights)
    end
    TestUtils.test_inverse_model(inverse_kkt_model, observation_model, ibr_nash.x, ibr_nash.x)
end

# TODO: does not reliably converge yet
@test_broken false && begin
    inverse_ibr_converged, inverse_ibr_solution, inverse_ibr_models, inverse_ibr_player_weights =
        solve_inverse_game(
            InverseIBRSolver(),
            ibr_nash.x;
            observation_model,
            control_system,
            player_cost_models,
            u_init = ibr_nash.u,
        )
end
