const project_root_dir = realpath(joinpath(@__DIR__, ".."))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "test/utils"))))

import Random
import Statistics
import LinearAlgebra
import Distances
import ElectronDisplay
import VegaLite

import CollisionAvoidanceGame
import TestDynamics
import TestUtils
using JuMPOptimalControl.TrajectoryVisualization:
    visualize_trajectory, visualize_trajectory_batch, VegaLiteBackend
using JuMPOptimalControl.ForwardGame: IBRGameSolver, KKTGameSolver, solve_game
using JuMPOptimalControl.InverseGames:
    InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game
using Test: @test, @testset

# Utils
include("./utils.jl")

#==================================== Forward Game Formulation =====================================#

T = 20
Δt = 1.0
rng = Random.MersenneTwister(1)

control_system = TestDynamics.ProductSystem([
    TestDynamics.Unicycle(Δt),
    TestDynamics.Unicycle(Δt),
    TestDynamics.Unicycle(Δt),
    TestDynamics.Unicycle(Δt),
    TestDynamics.Unicycle(Δt),
])

player_configurations = [
    # Vehicle on the right lane wishing to merge left to go faster
    (;
        initial_lane = 1.0,
        initial_progress = 0,
        initial_speed = 0.2,
        target_speed = 0.3,
        speed_cost = 1.0,
        target_lane = 0.0,
        prox_cost = 0.3,
    ),
    # Fast vehicle from the back that would like to maintain its speed.
    (;
        initial_lane = 0,
        initial_progress = -3.0,
        initial_speed = 0.4,
        target_speed = 0.4,
        target_lane = 0.0,
        speed_cost = 1.0,
        prox_cost = 0.3,
    ),
    # Slow truck on the right lane
    (;
        initial_lane = 1.0,
        initial_progress = 2,
        initial_speed = 0.10,
        target_speed = 0.10,
        speed_cost = 1.0,
        target_lane = 1.0,
        prox_cost = 0.05,
    ),
    # Slow truck on the right lane
    (;
        initial_lane = 1.0,
        initial_progress = 4,
        initial_speed = 0.10,
        target_speed = 0.10,
        speed_cost = 1.0,
        target_lane = 1.0,
        prox_cost = 0.05,
    ),
    # Fast vehicle on the left lane wishing to merge back on the right lane and slow down
    (;
        initial_lane = 0.0,
        initial_progress = 5,
        initial_speed = 0.3,
        target_speed = 0.2,
        speed_cost = 1.0,
        target_lane = 1.0,
        prox_cost = 0.3,
    ),
]

x0 = mapreduce(vcat, player_configurations) do player_config
    [
        player_config.initial_lane,
        player_config.initial_progress,
        player_config.initial_speed,
        deg2rad(90),
    ]
end

player_cost_models_gt = map(Iterators.countfrom(1), player_configurations) do ii, player_config
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model(;
        player_idx = ii,
        control_system,
        T,
        goal_position = nothing,
        weights = (;
            state_proximity = player_config.prox_cost,
            state_velocity = player_config.speed_cost,
            control_Δv = 1,
            control_Δθ = 1,
        ),
        y_lane_center = player_config.target_lane,
        target_speed = player_config.target_speed,
    )
end

converged_gt, forward_solution_gt, forward_opt_model_gt = solve_game(
    IBRGameSolver(),
    control_system,
    player_cost_models_gt,
    x0,
    T;
    ibr_convergence_tolerance = 1e-8,
    solver_attributes = (; print_level = 1),
)

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

viz = let
    max_size = 500
    y_position_domain = [-3, 13]
    x_position_domain = [-1, 2]
    x_range = only(diff(extrema(x_position_domain) |> collect))
    y_range = only(diff(extrema(y_position_domain) |> collect))
    max_range = max(x_range, y_range)
    canvas = VegaLite.@vlplot(
        width = max_size * x_range / max_range,
        height = max_size * y_range / max_range
    )

    subsampled_taj = forward_solution_gt_kkt.x[:, 1:end]

    visualize_trajectory(
        control_system,
        subsampled_taj,
        VegaLiteBackend();
        x_position_domain,
        y_position_domain,
        canvas,
    )
end

display(viz)

# TODO: continue here: player 3 and four need proximity cost or must start at a velocity that is not
# their target speed so that the speed cost remains observable.
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
