using Test: @test, @testset

using PartiallyObservedInverseGames.TrajectoryVisualization: visualize_trajectory
using PartiallyObservedInverseGames.ForwardGame: IBRGameSolver, KKTGameSolver, solve_game
using PartiallyObservedInverseGames.InverseGames:
    InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game

unique!(push!(LOAD_PATH, joinpath(@__DIR__, "utils")))
import TestUtils
import TestDynamics
import CollisionAvoidanceGame

control_system =
    TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), TestDynamics.Unicycle(0.25)])

x0 = vcat([-1, 0, 0.1, 0 + deg2rad(10)], [0, -1, 0.1, pi / 2 + deg2rad(10)])
T = 25
player_cost_models = let
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model(;
        player_idx = 1,
        control_system,
        T,
        goal_position = [1, 0],
    )
    cost_model_p2 = CollisionAvoidanceGame.generate_player_cost_model(;
        player_idx = 2,
        control_system,
        T,
        goal_position = [0, 1],
    )

    (cost_model_p1, cost_model_p2)
end

@testset "Product Dynamics" begin
    @test control_system.n_states == 8
    @test control_system.n_controls == 4

    @test all(i in TestDynamics.state_indices(control_system, 1) for i in 1:4)
    @test all(i in TestDynamics.state_indices(control_system, 2) for i in 5:8)
    @test all(i in TestDynamics.input_indices(control_system, 1) for i in 1:2)
    @test all(i in TestDynamics.input_indices(control_system, 2) for i in 3:4)
end

@testset "Forward Game" begin
    @testset "IBR" begin
        ibr_converged, ibr_solution, ibr_models =
            solve_game(IBRGameSolver(), control_system, player_cost_models, x0, T)
        global ibr_solution

        @test ibr_converged
    end

    @testset "KKT" begin
        kkt_converged, kkt_solution, kkt_model = solve_game(
            KKTGameSolver(),
            control_system,
            player_cost_models,
            x0,
            T;
            init = ibr_solution,
        )
        global kkt_solution

        @test kkt_converged
    end
end

@testset "Inverse Game" begin
    @testset "Perfect Observation" begin
        observation_model = (; σ = 0, expected_observation = identity)

        @testset "Inverse KKT Constraints" begin
            converged, inverse_kkt_solution, inverse_kkt_model = solve_inverse_game(
                InverseKKTConstraintSolver(),
                kkt_solution.x;
                control_system,
                observation_model,
                player_cost_models,
            )

            @test converged

            for (cost_model, weights) in
                zip(player_cost_models, inverse_kkt_solution.player_weights)
                TestUtils.test_inverse_solution(weights, cost_model.weights)
            end

            TestUtils.test_inverse_model(
                inverse_kkt_model,
                observation_model,
                kkt_solution.x,
                kkt_solution.x,
            )
        end

        @testset "Invsere KKT Residuals" begin
            converged, inverse_kkt_solution, inverse_kkt_model = solve_inverse_game(
                InverseKKTResidualSolver(),
                kkt_solution.x,
                kkt_solution.u;
                control_system,
                player_cost_models,
            )

            @test converged

            for (cost_model, weights) in
                zip(player_cost_models, inverse_kkt_solution.player_weights)
                TestUtils.test_inverse_solution(weights, cost_model.weights)
            end

            TestUtils.test_inverse_model(
                inverse_kkt_model,
                observation_model,
                kkt_solution.x,
                kkt_solution.x,
            )
        end
    end

    @testset "Noisy Full Observation" begin
        observation_model = (; σ = 0.01, expected_observation = identity)
        y_obs = TestUtils.noisy_observation(observation_model, kkt_solution.x)

        converged, inverse_kkt_solution, inverse_kkt_model = solve_inverse_game(
            InverseKKTConstraintSolver(),
            y_obs;
            control_system,
            observation_model,
            player_cost_models,
        )

        @test converged

        TestUtils.test_inverse_model(inverse_kkt_model, observation_model, kkt_solution.x, y_obs)
    end

    @testset "Noise-Free Partial Observation" begin
        observation_model = (; σ = 0.0, expected_observation = x -> x[[1, 2, 4, 5, 6, 8], :])
        y_obs = TestUtils.noisy_observation(observation_model, kkt_solution.x)

        converged, inverse_kkt_solution, inverse_kkt_model = solve_inverse_game(
            InverseKKTConstraintSolver(),
            y_obs;
            control_system,
            observation_model,
            player_cost_models,
            init_with_observation = true,
            max_observation_error = 0.1,
        )

        @test converged

        TestUtils.test_inverse_model(inverse_kkt_model, observation_model, kkt_solution.x, y_obs)
    end

    @testset "Noisy Partial Observation" begin
        observation_model = (; σ = 0.01, expected_observation = x -> x[[1, 2, 4, 5, 6, 8], :])
        y_obs = TestUtils.noisy_observation(observation_model, kkt_solution.x)

        converged, inverse_kkt_solution, inverse_kkt_model = solve_inverse_game(
            InverseKKTConstraintSolver(),
            y_obs;
            control_system,
            observation_model,
            player_cost_models,
            init_with_observation = true,
            max_observation_error = 0.1,
        )

        @test converged

        TestUtils.test_inverse_model(inverse_kkt_model, observation_model, kkt_solution.x, y_obs)
    end
end
