using Test: @test, @testset

using JuMP: @objective, @variable, @NLconstraint, @NLexpression
using JuMPOptimalControl.DynamicsModelInterface: visualize_trajectory
using JuMPOptimalControl.ForwardGame: IBRGameSolver, KKTGameSolver, solve_game
using JuMPOptimalControl.InverseGames: InverseIBRSolver, InverseKKTSolver, solve_inverse_game

unique!(push!(LOAD_PATH, joinpath(@__DIR__, "utils")))
import TestUtils
import TestDynamics
using CostUtils: symbol

control_system = TestDynamics.ProductSystem([TestDynamics.Unicycle(), TestDynamics.Unicycle()])

@testset "Product Dynamics" begin
    @test control_system.n_states == 8
    @test control_system.n_controls == 4

    @test all(i in TestDynamics.state_indices(control_system, 1) for i in 1:4)
    @test all(i in TestDynamics.state_indices(control_system, 2) for i in 5:8)
    @test all(i in TestDynamics.input_indices(control_system, 1) for i in 1:2)
    @test all(i in TestDynamics.input_indices(control_system, 2) for i in 3:4)
end

# TODO We should probably pass the `ProductSystem` and the player index here to compute the
# `state_indices`, `input_indices`, and `opponent_indices`.
function generate_player_cost_model(;
    state_indices,
    input_indices,
    goal_position,
    weights = (;
        state_goal = 1,
        # state_proximity = 0.0,
        state_velocity = 10,
        control_Δv = 100,
        control_Δθ = 10,
    ),
    prox_min_regularization = 0.1,
    T_activate_goalcost = 100,
    # TODO: dirty hack
    opponent_indices = 1 in state_indices ? (5:8) : (1:4),
)
    function add_regularized_squared_distance!(opt_model, x_sub_ego, x_sub_opp)
        @NLexpression(
            opt_model,
            [t = 1:T],
            (x_sub_ego[1, t] - x_sub_opp[1, t])^2 +
            (x_sub_ego[2, t] - x_sub_opp[2, t])^2 +
            prox_min_regularization
        )
    end

    function add_objective!(opt_model, x, u; weights)
        T = size(x, 2)
        # TODO: Currently, this implementation is *not* agnostic to the order of players or
        # input and state dimensions of other players subsystems. Get these values from
        # somewhere else to make it agnostic:
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views x_sub_opp = x[opponent_indices, :]

        prox_cost = let
            d_sq = add_regularized_squared_distance!(opt_model, x_sub_ego, x_sub_opp)
            prox_cost = @variable(opt_model, prox_cost[t = 1:T])
            @NLconstraint(opt_model, [t = 1:T], prox_cost[t] == -log(d_sq[t]))
            prox_cost
        end

        J̃ = (;
            state_goal = sum(el -> el^2, x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position),
            state_proximity = sum(prox_cost),
            state_velocity = sum(el -> el^2, x_sub_ego[3, :]),
            control_Δv = sum(el -> el^2, u_sub_ego[1, :]),
            control_Δθ = sum(el -> el^2, u_sub_ego[2, :]),
        )
        @objective(opt_model, Min, sum(weights[k] * J̃[k] for k in keys(weights)))
    end

    function add_objective_gradients!(opt_model, x, u; weights)
        n_states, T = size(x)
        n_controls = size(u, 1)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views x_sub_opp = x[opponent_indices, :]

        dprox_dxy = let
            d_sq = add_regularized_squared_distance!(opt_model, x_sub_ego, x_sub_opp)
            dproxdx = @variable(opt_model, [t = 1:T])
            @NLconstraint(
                opt_model,
                [t = 1:T],
                dproxdx[t] == -2 * (x_sub_ego[1, t] - x_sub_opp[1, t]) / d_sq[t]
            )
            dproxdy = @variable(opt_model, [t = 1:T])
            @NLconstraint(
                opt_model,
                [t = 1:T],
                dproxdy[t] == -2 * (x_sub_ego[2, t] - x_sub_opp[2, t]) / d_sq[t]
            )
            [dproxdx'; dproxdy']
        end

        # TODO: Technically this is missing the negative gradient on the opponents state but we
        # can't control that anyway (certainly not in OL Nash). Must be fixed for non-decoupled
        # systems and potentially FB Nash.
        dJdx = let
            dJ̃dx_sub = (;
                state_goal = [
                    zeros(2, T_activate_goalcost - 1) (x_sub_ego[1:2, T_activate_goalcost:T].-goal_position)
                    zeros(2, T)
                ],
                state_proximity = [dprox_dxy; zeros(2, T)],
                state_velocity = [zeros(2, T); x_sub_ego[3, :]'; zeros(1, T)],
                control_Δv = zeros(size(x_sub_ego)),
                control_Δθ = zeros(size(x_sub_ego)),
            )
            dJdx_sub = sum(weights[k] * dJ̃dx_sub[symbol(k)] for k in keys(weights))
            [
                zeros(first(state_indices) - 1, T)
                dJdx_sub
                zeros(n_states - last(state_indices), T)
            ]
        end

        dJdu = let
            dJdu_sub = 2 * [weights[:control_Δv], weights[:control_Δθ]] .* u_sub_ego
            [
                zeros(first(input_indices) - 1, T)
                dJdu_sub
                zeros(n_controls - last(input_indices), T)
            ]
        end

        (; dx = dJdx, du = dJdu)
    end

    (; player_inputs = input_indices, weights, add_objective!, add_objective_gradients!)
end

player_cost_models = let
    cost_model_p1 = generate_player_cost_model(;
        state_indices = 1:4,
        input_indices = 1:2,
        goal_position = [1, 0],
    )
    cost_model_p2 = generate_player_cost_model(;
        state_indices = 5:8,
        input_indices = 3:4,
        goal_position = [0, 1],
    )

    (cost_model_p1, cost_model_p2)
end

observation_model = (; σ = 0, expected_observation = identity)
x0 = vcat([-1, 0, 0, 0 + deg2rad(30)], [0, -1, 0, pi / 2 + deg2rad(30)])
T = 100

@testset "Forward Game" begin
    @testset "IBR" begin
        global ibr_converged, ibr_solution, ibr_models =
            solve_game(IBRGameSolver(), control_system, player_cost_models, x0, T)

        @test ibr_converged
    end

    @testset "KKT" begin
        global kkt_solution, kkt_model = solve_game(
            KKTGameSolver(),
            control_system,
            player_cost_models,
            x0,
            T;
            init = ibr_solution,
        )
    end
end

@testset "Inverse Game" begin
    @testset "KKT" begin
        global inverse_kkt_solution, inverse_kkt_model = solve_inverse_game(
            InverseKKTSolver(),
            kkt_solution.x;
            control_system,
            player_cost_models
        )

        for (cost_model, weights) in zip(player_cost_models, inverse_kkt_solution.player_weights)
            TestUtils.test_inverse_solution(weights, cost_model.weights)
        end

        TestUtils.test_inverse_model(
            inverse_kkt_model,
            observation_model,
            ibr_solution.x,
            ibr_solution.x,
        )
    end
end
