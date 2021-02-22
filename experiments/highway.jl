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
using JuMPOptimalControl.TrajectoryVisualization:
    visualize_trajectory, visualize_trajectory_batch, VegaLiteBackend
using JuMPOptimalControl.ForwardGame: IBRGameSolver, solve_game

# Utils
include("./utils.jl")

#==================================== Forward Game Formulation =====================================#

T = 25
Δt = 0.25

control_system = TestDynamics.ProductSystem([
    TestDynamics.Unicycle(Δt),
    TestDynamics.Unicycle(Δt),
    TestDynamics.Unicycle(Δt),
])

# TODO: next
# - [done] remove proximity cost for slow vehicle
# - add separate initial velocity
# - remove goal position and add lane and nominal velocity instead
# - consider
player_configurations = [
    (;
        initial_speed = 0.2,
        initial_progress = 0,
        initial_lane = 0,
        target_speed = 1,
        goal_lane = 0.0,
        prox_cost = 1,
    ),
    (;
        initial_speed = 0.2,
        initial_progress = 0.5,
        initial_lane = -1,
        target_speed = 0.5,
        goal_lane = -1,
        prox_cost = 1,
    ),
    (;
        initial_speed = 0.2,
        initial_progress = 1.0,
        initial_lane = -1,
        target_speed = 0.2,
        goal_lane = -1,
        prox_cost = 0,
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
        goal_position = [
            player_config.goal_lane,
            player_config.initial_progress + Δt * T * player_config.target_speed
        ],
        weights = merge(
            (; state_proximity = 1, state_velocity = 1, control_Δv = 1, control_Δθ = 1),
            (; state_proximity = player_config.prox_cost),
        ),
    )
end

converged_gt, forward_solution_gt, forward_opt_model_gt = solve_game(
    IBRGameSolver(),
    control_system,
    player_cost_models_gt,
    x0,
    T;
    solver_attributes = (; print_level = 1),
)

visualize_trajectory(
    control_system,
    forward_solution_gt.x,
    VegaLiteBackend();
    canvas = VegaLite.@vlplot(width = 500, height = 500),
)
