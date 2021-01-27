import JuMP
import Ipopt
import Zygote

using JuMP: @variable, @constraint, @objective
using JuMPOptimalControl.ForwardGame: IBRGameSolver, KKTGameSolver, solve_game
using JuMPOptimalControl.InverseGames: solve_inverse_game
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

control_system = (
    add_dynamics_constraints! = add_unicycle_dynamics_constraints!,
    add_dynamics_jacobians! = add_unicycle_dynamics_jacobians!,
    n_states = 4,
    n_controls = 2,
)

x0 = [-1, 1, 0.1, 0]
T = 100

player_cost_models = (
    (;
        weights = (; state_velocity_p1 = 10, control_Δv_p1 = 100),
        objective = objective_p1,
        objective_gradients = objective_gradients_p1,
        # TODO: redundant. Handle in target function.
        add_objective! = function (model, args...; kwargs...)
            @objective(model, Min, objective_p1(args...; kwargs...))
        end,
        player_inputs = [1],
    ),
    (;
        weights = (; state_goal_p2 = 0.1, control_Δθ_p2 = 10),
        objective = objective_p2,
        objective_gradients = objective_gradients_p2,
        # TODO: redundant. Handle in target function.
        add_objective! = function (model, args...; kwargs...)
            @objective(model, Min, objective_p2(args...; kwargs...))
        end,
        player_inputs = [2],
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

function sanity_check_unicycle_multipliers(λ, x, u; player_cost_models)

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

            @test all(all(isapprox.(x, 0; atol = 1e-8)) for x in dLdx)
            @test all(all(isapprox.(x, 0; atol = 1e-8)) for x in dLdu)
        end
    end
end

@testset "Forward IBR" begin
    global ibr_nash, ibr_converged, ibr_models = solve_game(
        IBRGameSolver(),
        control_system,
        player_cost_models,
        x0,
        T;
        inner_solver_kwargs = (; silent = true),
    )
    @test ibr_converged

    # extract constraint multipliers
    global λ_ibr = mapreduce((a, b) -> cat(a, b; dims = 3), ibr_models) do model
        mapreduce(hcat, model[:dynamics]) do c
            # Sign flipped due to internal convention of JuMP
            -JuMP.dual.(c)
        end
    end

    sanity_check_unicycle_multipliers(λ_ibr, ibr_nash.x, ibr_nash.u; player_cost_models)
end

@testset "Forward KKT Nash" begin
    global kkt_nash, kkt_model = solve_game(
        KKTGameSolver(),
        control_system,
        player_cost_models,
        x0,
        T;
        init = (; x = ibr_nash.x, u = ibr_nash.u),
        solver = Ipopt.Optimizer,
    )

    visualize_unicycle_trajectory(kkt_nash.x)

    sanity_check_unicycle_multipliers(kkt_nash.λ, kkt_nash.x, kkt_nash.u; player_cost_models)
end

# TODO: robustify
@testset "Inverse KKT Nash" begin
    global inverse_kkt_solution, inverse_kkt_model = solve_inverse_game(
        kkt_nash.x,
        kkt_nash.u;
        # λ_init = kkt_nash.λ,
        control_system,
        player_cost_models,
    )
end
