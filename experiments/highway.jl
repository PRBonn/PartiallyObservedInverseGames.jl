const project_root_dir = realpath(joinpath(@__DIR__, ".."))
include("utils/preamble.jl")
load_cache_if_not_defined!("highway")

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
        initial_progress = 6,
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

position_indices = mapreduce(vcat, eachindex(control_system.subsystems)) do subsystem_idx
    TestDynamics.state_indices(control_system, subsystem_idx)[1:2]
end

partial_state_indices = mapreduce(vcat, eachindex(control_system.subsystems)) do subsystem_idx
    TestDynamics.state_indices(control_system, subsystem_idx)[[1, 2, 4]]
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

#===================================== Additional Visualization ====================================#

function visualize_highway(x; subsampling = 1, kwargs...)
    viz = let
        max_size = 500
        y_position_domain = [-4, 14]
        x_position_domain = [-1, 2]
        x_range = only(diff(extrema(x_position_domain) |> collect))
        y_range = only(diff(extrema(y_position_domain) |> collect))
        max_range = max(x_range, y_range)
        canvas = VegaLite.@vlplot(
            width = max_size * x_range / max_range,
            height = max_size * y_range / max_range
        )

        subsampled_taj = x[:, 1:subsampling:end]

        visualize_trajectory(
            control_system,
            subsampled_taj;
            x_position_domain,
            y_position_domain,
            canvas,
            kwargs...,
        )
    end
end

#======================================== Monte Carlo Study ========================================#

include("utils/monte_carlo_study.jl")

## Extra Visualization
@saveviz highway_frontfig_cost1 = CostHeatmapVisualizer.cost_viz(
    1,
    player_configurations[1];
    x_sequence = dataset[begin].ground_truth.x[:, 1:1],
    control_system,
)
@saveviz highway_frontfig_cost5 = CostHeatmapVisualizer.cost_viz(
    5,
    player_configurations[5];
    x_sequence = dataset[begin].ground_truth.x[:, 1:1],
    control_system,
)
@saveviz highway_frontfig_gt_viz = visualize_highway(dataset[begin].ground_truth.x)
@saveviz highway_frontfig_observation_viz =
    visualize_highway(dataset[end].ground_truth.x; draw_line = false)
