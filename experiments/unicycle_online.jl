const project_root_dir = realpath(joinpath(@__DIR__, ".."))
include("preamble.jl")
load_cache_if_not_defined!("unicycle_online")

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
        T_activate_goalcost = 1,
        fix_costs = (; state_goal = 0.1),
    )
end

#======================================== Monte Carlo Study ========================================#

## Dataset Generation
# TODO restore the original number of samples (40)
n_observation_sequences_per_instance = 5
T_predict = 10
observation_horizons = 5:(T - T_predict)

dataset = MonteCarloStudy.generate_dataset_observation_window_sweep(;
    observation_horizons,
    noise_level = 0.05,
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
    player_weight_prior = nothing,
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

frame = [-floor(n_observation_sequences_per_instance), 0]
@saveviz parameter_error_viz =
    errstats |> MonteCarloStudy.visualize_paramerr_over_obshorizon(; frame, round_x_axis = false)
@saveviz position_error_viz =
    errstats |> MonteCarloStudy.visualize_poserr_over_obshorizon(; frame, round_x_axis = false)

@saveviz prediction_comparison_viz = let
    opacity = 0.5
    d = dataset[30]
    window = 1:(d.observation_horizon + T_predict)
    ground_truth =
        TrajectoryVisualization.trajectory_data(control_system, d.ground_truth.x[:, window])
    observation = TrajectoryVisualization.trajectory_data(control_system, d.observation.x)

    ours = let
        estimate = estimates_conKKT[d.idx]
        @assert estimate.idx == d.idx
        TrajectoryVisualization.trajectory_data(control_system, estimate.estimate.x)
    end

    baseline = let
        estimate = estimates_resKKT[d.idx]
        @assert estimate.idx == d.idx
        TrajectoryVisualization.trajectory_data(control_system, estimate.estimate.x)
    end

    canvas = VegaLite.@vlplot(title = "Observation Horizon: $(d.observation_horizon)")

    canvas = TrajectoryVisualization.visualize_trajectory(
        ours;
        group = "ours",
        legend = true,
        opacity,
        canvas,
    )

    canvas = TrajectoryVisualization.visualize_trajectory(
        baseline;
        group = "baseline",
        legend = true,
        opacity,
        canvas,
    )

    canvas = TrajectoryVisualization.visualize_trajectory(
        ground_truth;
        group = "ground truth",
        legend = true,
        opacity,
        canvas,
    )

    canvas = TrajectoryVisualization.visualize_trajectory(
        observation;
        group = "observation",
        legend = true,
        draw_line = false,
        opacity,
        canvas,
    )
end
