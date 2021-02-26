const project_root_dir = realpath(joinpath(@__DIR__, ".."))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "experiments/MonteCarloStudy"))))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "test/utils"))))

import Distributed
Distributed.@everywhere begin
    import Pkg
    Pkg.activate($project_root_dir)
    union!(LOAD_PATH, $LOAD_PATH)

    import MonteCarloStudy
    import CollisionAvoidanceGame
    import TestDynamics
    using JuMPOptimalControl.ForwardGame: IBRGameSolver, KKTGameSolver
    using JuMPOptimalControl.InverseGames: InverseKKTConstraintSolver, InverseKKTResidualSolver

end

import ElectronDisplay
import VegaLite
import Random
using JuMPOptimalControl.TrajectoryVisualization: VegaLiteBackend, visualize_trajectory

# Utils
include("utils/distributed.jl")
include("utils/misc.jl")
include("utils/simple_caching.jl")
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
        y_position_domain = [-3, 13]
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
            subsampled_taj,
            VegaLiteBackend();
            x_position_domain,
            y_position_domain,
            canvas,
            kwargs...
        )
    end
end

#======================================== Monte Carlo Study ========================================#

## Dataset Generation
n_observation_sequences_per_noise_level = 40

#TODO run_cached needs experiments prefix
@run_cached forward_solution_gt, dataset = MonteCarloStudy.generate_dataset(;
    solve_args = (; solver = IBRGameSolver(), control_system, player_cost_models_gt, x0, T),
    noise_levels = unique([0:0.001:0.01; 0.01:0.005:0.03; 0.03:0.01:0.1]),
    n_observation_sequences_per_noise_level,
)

## Estimation
estimator_setup = (;
    dataset,
    control_system,
    player_cost_models = player_cost_models_gt,
    solver_attributes = (; print_level = 1),
)
estimator_setup_partial =
    merge(estimator_setup, (; expected_observation = x -> x[partial_state_indices, :]))
@run_cached estimates_conKKT =
    MonteCarloStudy.estimate(InverseKKTConstraintSolver(); estimator_setup...)
@run_cached estimates_conKKT_partial =
    MonteCarloStudy.estimate(InverseKKTConstraintSolver(); estimator_setup_partial...)
@run_cached estimates_resKKT =
    MonteCarloStudy.estimate(InverseKKTResidualSolver(); estimator_setup...)
@run_cached estimates_resKKT_partial =
    MonteCarloStudy.estimate(InverseKKTResidualSolver(); estimator_setup_partial...)

## Forward Solution Augmentation
augmentor_kwargs = (;
    solver = KKTGameSolver(),
    control_system,
    player_cost_models_gt,
    x0,
    T,
    match_equilibrium = (; forward_solution_gt.x),
    init = (; forward_solution_gt.x, forward_solution_gt.u),
    solver_attributes = (; print_level = 1),
)
@run_cached augmented_estimates_resKKT =
    MonteCarloStudy.augment_with_forward_solution(estimates_resKKT; augmentor_kwargs...)
@run_cached augmented_estimates_resKKT_partial =
    MonteCarloStudy.augment_with_forward_solution(estimates_resKKT_partial; augmentor_kwargs...)
estimates = [
    estimates_conKKT
    estimates_conKKT_partial
    augmented_estimates_resKKT
    augmented_estimates_resKKT_partial
]

## Error Ststistics Computation
demo_gt = merge((; player_cost_models_gt), forward_solution_gt)
errstats = map(estimates) do estimate
    MonteCarloStudy.estimator_statistics(estimate; dataset, demo_gt, position_indices)
end

## Visualization

frame = [-2n_observation_sequences_per_noise_level, 0]
@saveviz parameter_error_viz = errstats |> MonteCarloStudy.visualize_paramerr(; frame)
@saveviz position_error_viz = errstats |> MonteCarloStudy.visualize_poserr(; frame)
