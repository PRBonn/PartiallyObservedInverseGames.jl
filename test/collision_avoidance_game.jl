using Test: @test, @testset

using JuMP: @objective, @variable, @NLconstraint, @NLexpression
using JuMPOptimalControl.DynamicsModelInterface: visualize_trajectory
using JuMPOptimalControl.ForwardGame: IBRGameSolver, KKTGameSolver, solve_game
using JuMPOptimalControl.InverseGames: InverseIBRSolver, InverseKKTSolver, solve_inverse_game

unique!(push!(LOAD_PATH, joinpath(@__DIR__, "utils")))
import TestUtils
import TestDynamics

control_system = TestDynamics.ProductSystem([TestDynamics.Unicycle(), TestDynamics.Unicycle()])

@testset "Product Dynamics" begin
    @test control_system.n_states == 8
    @test control_system.n_controls == 4

    @test all(i in TestDynamics.state_indices(control_system, 1) for i in 1:4)
    @test all(i in TestDynamics.state_indices(control_system, 2) for i in 5:8)
    @test all(i in TestDynamics.input_indices(control_system, 1) for i in 1:2)
    @test all(i in TestDynamics.input_indices(control_system, 2) for i in 3:4)
end

function generate_player_cost_model(;
    state_indices,
    input_indices,
    goal_position,
    weights = (; state_goal = 1, state_proximity = 0.01, state_velocity = 10, control_Δv = 100, control_Δθ = 10),
    prox_min_regularization = 0.1,
    T_activate_goalcost = 100,
)

    function add_objective!(opt_model, x, u; weights)
        T = size(x, 2)
        # TODO: Currently, this implementation is *not* agnostic to the order of players or
        # input and state dimensions of other players subsystems. Get these values from
        # somewhere else to make it agnostic:
        @views x_sub = x[state_indices, :]
        @views u_sub = u[input_indices, :]

        prox_cost = let
            regularized_sq_dist = let
                opponent_position_indices = 1 in state_indices ? (5:6) : (1:2)
                @NLexpression(
                    opt_model,
                    [t = 1:T],
                    (x_sub[1, t] - x[opponent_position_indices[1], t])^2 +
                    (x_sub[2, t] - x[opponent_position_indices[2], t])^2 +
                    prox_min_regularization
                )
            end

            prox_cost = @variable(opt_model, prox_cost[t = 1:T])
            @NLconstraint(opt_model, [t = 1:T], prox_cost[t] == -log(regularized_sq_dist[t]))
            prox_cost
        end

        J̃ = (;
            state_goal = sum(el -> el^2, x_sub[1:2, T_activate_goalcost:T] .- goal_position),
            state_proximity = sum(prox_cost),
            state_velocity = sum(el -> el^2, x_sub[3, :]),
            control_Δv = sum(el -> el^2, u_sub[1, :]),
            control_Δθ = sum(el -> el^2, u_sub[2, :]),
        )
        @objective(opt_model, Min, sum(weights[k] * J̃[k] for k in keys(weights)))
    end

    function add_objective_gradients!(opt_model, x, u; weights)
        n_states, T = size(x)
        n_controls = size(u, 1)
        @views x_sub = x[state_indices, :]
        @views u_sub = u[input_indices, :]

        dJdx_sub = [
            2 * weights[:state_goal] * (x_sub[1:2, :] .- goal_position)
            2 * weights[:state_velocity] * x_sub[3, :]'
            zeros(1, T)
        ]
        dJdu_sub = 2 .* [weights[:control_Δv], weights[:control_Δθ]] .* u_sub

        dJdx = [
            zeros(first(state_indices) - 1, T)
            dJdx_sub
            zeros(n_states - last(state_indices), T)
        ]
        dJdu = [
            zeros(first(input_indices) - 1, T)
            dJdu_sub
            zeros(n_controls - last(input_indices), T)
        ]

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

false && @testset "Inverse Game" begin
    @testset "KKT" begin
        global inverse_kkt_solution, inverse_kkt_model = solve_inverse_game(
            InverseKKTSolver(),
            ibr_solution.x;
            control_system,
            player_cost_models,
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
