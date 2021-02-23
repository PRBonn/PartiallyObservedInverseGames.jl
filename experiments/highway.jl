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

T = 40
Δt = 0.5
rng = Random.MersenneTwister(1)

control_system = TestDynamics.ProductSystem([
    TestDynamics.Unicycle(Δt),
    TestDynamics.Unicycle(Δt),
    TestDynamics.Unicycle(Δt),
    TestDynamics.Unicycle(Δt),
    TestDynamics.Unicycle(Δt),
])

# TODO: next
# - [done] remove proximity cost for slow vehicle
# - [done] add lane cost
# - [done] add separate initial velocity
# - [done] remove goal position and add lane and nominal velocity instead
# - [done] DFK:
#   - [done] quadratic penalty to the *target lane*
#   - [done] start off the target lane
#   - [done] have objective to go at a specific speed
#   - [done] remove log-barriers
#  - [done] try different IBR orders.
#  - [done] add antother player merging from the left to the right
#  - [done] tidy up parameterization of CollisionAvoidanceGame.
#  - implement gradients for additional cost terms
#  - test gradient with forward solver against IBR
#  - Figure out which parameters are worth inferring here.
#  - make sure that the old example still works.
#
#  Later: vary some parameter dimensions
#   - fix initial progress, initial speed, initial lane, goal lane, target_speed
#   - vary prox cost (in low regime near 0.0 to 0.3), speed cost (in high regime near),
#
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
        initial_speed = 0.15,
        target_speed = 0.15,
        speed_cost = 1.0,
        target_lane = 1.0,
        prox_cost = 0.1,
    ),
    # Slow truck on the right lane
    (;
        initial_lane = 1.0,
        initial_progress = 4,
        initial_speed = 0.15,
        target_speed = 0.15,
        speed_cost = 1.0,
        target_lane = 1.0,
        prox_cost = 0.1,
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
    solver_attributes = (; print_level = 1),
)

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

    subsampled_taj = forward_solution_gt.x[:, 1:2:end]

    visualize_trajectory(
        control_system,
        subsampled_taj,
        VegaLiteBackend();
        x_position_domain,
        y_position_domain,
        canvas,
    )
end
