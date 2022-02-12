const project_root_dir = realpath(joinpath(@__DIR__, ".."))
include("utils/preamble.jl")
include("utils/unicycle_online_setup.jl")
load_cache_if_not_defined!("unicycle_online")

#======================================== Monte Carlo Study ========================================#

## Dataset Generation
n_observation_sequences_per_instance = 40
T_predict = 10
observation_horizons = 5:(T - T_predict)
noise_level = 0.05

@run_cached dataset = MonteCarloStudy.generate_dataset_observation_window_sweep(;
    observation_horizons,
    noise_level,
    solve_args = (; solver = IBRGameSolver(), control_system, player_cost_models_gt, x0, T),
    n_observation_sequences_per_instance,
)

## Estimation
estimator_setup = (;
    dataset,
    control_system,
    player_cost_models = player_cost_models_gt,
    solver_attributes = (; print_level = 1, max_iter = 500),
    T_predict,
    cmin = 1e-3,
    prior = nothing,
    pre_solve_kwargs = (; u_regularization = 1e-5),
    max_observation_error = nothing,
)
estimator_setup_partial =
    merge(estimator_setup, (; expected_observation = x -> x[partial_state_indices, :]))

@run_cached estimates_conKKT =
    MonteCarloStudy.estimate(InverseKKTConstraintSolver(); estimator_setup...)
@run_cached estimates_conKKT_partial =
    MonteCarloStudy.estimate(InverseKKTConstraintSolver(); estimator_setup_partial...)
@run_cached estimates_resKKT =
    MonteCarloStudy.estimate(AugmentedInverseKKTResidualSolver(); estimator_setup...)
@run_cached estimates_resKKT_partial =
    MonteCarloStudy.estimate(AugmentedInverseKKTResidualSolver(); estimator_setup_partial...)

estimates =
    [
        estimates_conKKT
        estimates_conKKT_partial
        estimates_resKKT
        estimates_resKKT_partial
    ] |> e -> filter(e -> e.converged, e)

errstats = map(estimates) do estimate
    MonteCarloStudy.estimator_statistics(
        estimate;
        player_cost_models_gt,
        position_indices,
        window_type = :prediction,
        T_predict,
    )
end

@saveviz parameter_error_viz = errstats |> MonteCarloStudy.visualize_paramerr_over_obshorizon()
@saveviz position_error_viz = errstats |> MonteCarloStudy.visualize_poserr_over_obshorizon()

@saveviz prediction_comparison_viz = let
    opacity = 0.6
    d = filter(d -> d.observation_horizon == 10, dataset)[1]
    window = 1:(d.observation_horizon + T_predict)
    ground_truth = let
        trajectory = TrajectoryVisualization.trajectory_data(
            control_system,
            d.ground_truth.x[:, window],
        )
        player_weights = [m.weights for m in player_cost_models_gt]
        (; player_weights, trajectory)
    end

    observation_trajectory =
        TrajectoryVisualization.trajectory_data(control_system, d.observation.x)

    ours = let
        estimate = estimates_conKKT_partial[d.idx]
        @assert estimate.idx == d.idx
        player_weights = estimate.estimate.player_weights
        trajectory =
            TrajectoryVisualization.trajectory_data(control_system, estimate.estimate.x)
        (; player_weights, trajectory)
    end

    baseline = let
        estimate = estimates_resKKT_partial[d.idx]
        @assert estimate.idx == d.idx
        player_weights = estimate.estimate.player_weights
        trajectory =
            TrajectoryVisualization.trajectory_data(control_system, estimate.estimate.x)
        (; player_weights, trajectory)
    end

    trajectory_canvas = VegaLite.@vlplot(width = 400, height = 400,)

    groups = ["Ground Truth", "Observation", "Ours Partial", "Baseline Partial"]
    color_scale = VegaLite.@vlfrag(
        domain = groups,
        range = [MonteCarloStudy.color_map[g] for g in groups]
    )

    trajectory_canvas = TrajectoryVisualization.visualize_trajectory(
        ours.trajectory;
        group = "Ours Partial",
        color_scale,
        legend = true,
        opacity,
        canvas = trajectory_canvas,
    )

    trajectory_canvas = TrajectoryVisualization.visualize_trajectory(
        baseline.trajectory;
        group = "Baseline Partial",
        color_scale,
        legend = true,
        opacity,
        canvas = trajectory_canvas,
    )

    trajectory_canvas = TrajectoryVisualization.visualize_trajectory(
        ground_truth.trajectory;
        group = "Ground Truth",
        color_scale,
        legend = true,
        opacity,
        canvas = trajectory_canvas,
    ) + VegaLite.@vlplot(
        data = filter(s -> s.t == 1, ground_truth.trajectory),
        mark = {"text", dx = -10, dy = 15},
        text = "player",
        x = "px:q",
        y = "py:q",
    )

    trajectory_canvas = TrajectoryVisualization.visualize_trajectory(
        observation_trajectory;
        group = "Observation",
        color_scale,
        legend = true,
        draw_line = false,
        opacity,
        canvas = trajectory_canvas,
    )

    cost_data =
        map(1:2) do ii
            player = "Player-$ii"
            [
                (;
                    player,
                    group = "Baseline Partial",
                    weight = baseline.player_weights[ii][:state_proximity],
                ), # TODO: get programatically
                (; player, group = "Ground Truth", weight = 0.25), # TODO: get programatically
                (;
                    player,
                    group = "Ours Partial",
                    weight = ours.player_weights[ii][:state_proximity],
                ),
            ]
        end |> d -> reduce(vcat, d)

    cost_canvas =
        cost_data |> VegaLite.@vlplot(
            mark = :bar,
            width = 400 / 2 - 10,
            height = 30,
            column = {"player", title = nothing},
            x = {"weight:q", title = "Proximity Cost Weight"},
            y = {"group:o", title = nothing},
            color = "group:o"
        )

    VegaLite.@vlplot(spacing = 10) + [trajectory_canvas; cost_canvas]
end
