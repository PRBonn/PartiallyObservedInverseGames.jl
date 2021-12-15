const project_root_dir = realpath(joinpath(@__DIR__, ".."))
include("utils/preamble.jl")
load_cache_if_not_defined!("unicycle")

import LazyGroupBy: grouped

#==================================== Forward Game Formulation =====================================#

T = 25

control_system =
    TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), TestDynamics.Unicycle(0.25)])

player_angles = let
    n_players = length(control_system.subsystems)
    map(eachindex(control_system.subsystems)) do ii
        angle_fraction = n_players == 2 ? pi / 2 : 2pi / n_players
        (ii - 1) * angle_fraction
    end
end

x0 = mapreduce(vcat, player_angles) do player_angle
    [unitvector(player_angle + pi); 0.1; player_angle + deg2rad(10)]
end

position_indices = mapreduce(vcat, eachindex(control_system.subsystems)) do subsystem_idx
    TestDynamics.state_indices(control_system, subsystem_idx)[1:2]
end

partial_state_indices = mapreduce(vcat, eachindex(control_system.subsystems)) do subsystem_idx
    TestDynamics.state_indices(control_system, subsystem_idx)[[1, 2, 4]]
end

player_cost_models_gt = map(enumerate(player_angles)) do (ii, player_angle)
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model(;
        player_idx = ii,
        control_system,
        T,
        goal_position = unitvector(player_angle),
    )
end

#======================================== Monte Carlo Study ========================================#

## Dataset Generation
n_observation_sequences_per_instance = 40

@run_cached dataset = MonteCarloStudy.generate_dataset_noise_sweep(;
    solve_args = (; solver = IBRGameSolver(), control_system, player_cost_models_gt, x0, T),
    noise_levels = unique([0:0.001:0.01; 0.01:0.005:0.03; 0.03:0.01:0.1]),
    n_observation_sequences_per_instance,
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
    solver_attributes = (; print_level = 1),
)
@run_cached augmented_estimates_resKKT =
    MonteCarloStudy.augment_with_forward_solution(estimates_resKKT; augmentor_kwargs...)
@run_cached augmented_estimates_resKKT_partial =
    MonteCarloStudy.augment_with_forward_solution(estimates_resKKT_partial; augmentor_kwargs...)

@save_json estimates = [
    estimates_conKKT
    estimates_conKKT_partial
    augmented_estimates_resKKT
    augmented_estimates_resKKT_partial
]

## Error Ststistics Computation
@save_json errstats = map(estimates) do estimate
    MonteCarloStudy.estimator_statistics(
        estimate;
        player_cost_models_gt,
        position_indices,
        window_type = :observation,
        T_predict = 0,
    )
end

## Visualization
demo_noise_level = 0.1
trajectory_viz_config = (;
    x_position_domain = (-1.2, 1.2),
    y_position_domain = (-1.2, 1.2),
    opacity = 0.5,
    legend = false,
)

@save_json trajectory_data_gt =
    TrajectoryVisualization.trajectory_data(control_system, dataset[begin].ground_truth.x)

@save_json trajectory_data_obs = [
    TrajectoryVisualization.trajectory_data(control_system, d.observation.x) for
    d in dataset if d.σ == demo_noise_level
]

@save_json trajectory_data_estimates =
    map.(
        e -> TrajectoryVisualization.trajectory_data(control_system, e.estimate.x),
        grouped(
            e -> e.estimator_name,
            Iterators.filter(e -> e.converged && e.σ == demo_noise_level, estimates),
        ),
    )

groups = [
    "Ground Truth",
    "Observation",
    "Ours Full",
    "Ours Partial",
    "Baseline Full",
    "Baseline Partial",
]
color_scale =
    VegaLite.@vlfrag(domain = groups, range = [MonteCarloStudy.color_map[g] for g in groups])

ground_truth_viz =
    TrajectoryVisualization.visualize_trajectory(
        trajectory_data_gt;
        canvas = VegaLite.@vlplot(width = 200, height = 200),
        legend = VegaLite.@vlfrag(orient = "top", offset = 5),
        trajectory_viz_config...,
        group = "Ground Truth",
        color_scale,
    ) + VegaLite.@vlplot(
        data = filter(s -> s.t == 1, trajectory_data_gt),
        mark = {"text", dx = 8, dy = 8},
        text = "player",
        x = "px:q",
        y = "py:q",
    )

observations_bundle_viz =
    VegaLite.@vlplot() + MonteCarloStudy.visualize_trajectory_batch(
        trajectory_data_obs;
        trajectory_viz_config...,
        draw_line = false,
        group = "Observation",
        color_scale,
    )

viz_trajectory_estiamtes = Dict(
    k => TrajectoryVisualization.visualize_trajectory_batch(
        v;
        trajectory_viz_config...,
        color_scale,
        group = k,
    ) for (k, v) in trajectory_data_estimates
)

@saveviz demo_trajs_viz = hcat(
    VegaLite.@vlplot(title = "Ground Truth") + ground_truth_viz,
    VegaLite.@vlplot(title = "Observations") + observations_bundle_viz,
)

@saveviz ours_trajs_viz = hcat(
    VegaLite.@vlplot(title = "Full Observation") + viz_trajectory_estiamtes["Ours Full"],
    VegaLite.@vlplot(title = "Partial Observation") + viz_trajectory_estiamtes["Ours Partial"],
)

@saveviz baseline_trajs_viz = hcat(
    VegaLite.@vlplot(title = "Full Observation") + viz_trajectory_estiamtes["Baseline Full"],
    VegaLite.@vlplot(title = "Partial Observation") + viz_trajectory_estiamtes["Baseline Partial"],
)

frame = [-floor(1.5n_observation_sequences_per_instance), 0]
@saveviz parameter_error_viz =
    errstats |> MonteCarloStudy.visualize_paramerr_over_noise(; frame, round_x_axis = false)
@saveviz position_error_viz =
    errstats |> MonteCarloStudy.visualize_poserr_over_noise(; frame, round_x_axis = false)
