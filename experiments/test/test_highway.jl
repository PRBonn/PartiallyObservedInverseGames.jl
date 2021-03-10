using Test: @test, @testset
import TestUtils

@testset "Gradient integration test" begin
    converged_gt_kkt, forward_solution_gt_kkt, forward_opt_model_gt_kkt = solve_game(
        KKTGameSolver(),
        control_system,
        player_cost_models_gt,
        x0,
        T;
        init = (; x = forward_solution_gt.x),
        solver_attributes = (; print_level = 3),
    )

    global forward_solution_gt_kkt

    @test converged_gt_kkt
    @test isapprox(forward_solution_gt_kkt.x, forward_solution_gt.x, atol = 1e-2)
    @test isapprox(forward_solution_gt_kkt.u, forward_solution_gt.u, atol = 1e-4)
end

@testset "Inverse solutions integration test" begin
    @testset "Residual baseline" begin
        # Minimal inverse test with both solvers:
        converged_res, estimate_res, opt_model_res = solve_inverse_game(
            InverseKKTResidualSolver(),
            forward_solution_gt.x,
            forward_solution_gt.u;
            control_system,
            player_cost_models = player_cost_models_gt,
        )

        @test converged_res
        for (cost_model, weights) in zip(player_cost_models_gt, estimate_res.player_weights)
            TestUtils.test_inverse_solution(weights, cost_model.weights)
        end
    end

    @testset "Constraint Solver" begin
        # Minimal inverse test with both solvers:
        converged_con, estimate_con, opt_model_con = solve_inverse_game(
            InverseKKTConstraintSolver(),
            forward_solution_gt.x;
            observation_model = (; σ = 0, expected_observation = identity),
            control_system,
            player_cost_models = player_cost_models_gt,
        )

        @test converged_con
        for (ii, cost_model, weights) in
            zip(Iterators.countfrom(), player_cost_models_gt, estimate_con.player_weights)
            println("Player $ii")
            TestUtils.test_inverse_solution(weights, cost_model.weights; verbose = true)
        end
    end
end
