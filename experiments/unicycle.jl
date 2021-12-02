const project_root_dir = realpath(joinpath(@__DIR__, ".."))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "experiments/utils/MonteCarloStudy"))))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "test/utils"))))

using Distributed: Distributed
Distributed.@everywhere begin
    using Pkg: Pkg
    Pkg.activate($project_root_dir)
    union!(LOAD_PATH, $LOAD_PATH)

    using MonteCarloStudy: MonteCarloStudy
    using CollisionAvoidanceGame: CollisionAvoidanceGame
    using TestDynamics: TestDynamics
    using PartiallyObservedInverseGames.ForwardGame: IBRGameSolver, KKTGameSolver
    using PartiallyObservedInverseGames.InverseGames:
        InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game
end

import PartiallyObservedInverseGames.TrajectoryVisualization
using VegaLite: VegaLite
import LazyGroupBy: grouped

# Utils
include("utils/misc.jl")
include("utils/simple_caching.jl")
load_cache_if_not_defined!("unicycle")

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
n_observation_sequences_per_noise_level = 40

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

# truncated horizon inference test
estimated_traj_data = let
    d = dataset[begin]
    observation_horizon = T ÷ 3
    y = d.x[:, 1:observation_horizon]
    observation_model = (; d.σ, expected_observation = identity)
    converged, sol = solve_inverse_game(
        InverseKKTConstraintSolver(),
        y;
        control_system,
        observation_model,
        player_cost_models = player_cost_models_gt,
        T,
    )
    @assert converged
    TrajectoryVisualization.trajectory_data(control_system, sol.x)
end

estimated_traj_data |> TrajectoryVisualization.visualize_trajectory

#==

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
@save_json errstats = map(estimates) do estimate
    MonteCarloStudy.estimator_statistics(estimate; dataset, demo_gt, position_indices)
end

## Visualization
demo_noise_level = 0.1
trajectory_viz_domain = (; x_position_domain = (-1.2, 1.2), y_position_domain = (-1.2, 1.2))

@save_json trajectory_data_gt =
    TrajectoryVisualization.trajectory_data(control_system, forward_solution_gt.x)

@save_json trajectory_data_obs = [
    TrajectoryVisualization.trajectory_data(control_system, d.x) for
    d in dataset if d.σ == demo_noise_level
]

@save_json trajectory_data_estimates =
    map.(
        e -> TrajectoryVisualization.trajectory_data(control_system, e.x),
        grouped(
            e -> e.estimator_name,
            Iterators.filter(
                e -> e.converged && dataset[e.observation_idx].σ == demo_noise_level,
                estimates,
            ),
        ),
    )

ground_truth_viz =
    TrajectoryVisualization.visualize_trajectory(
        trajectory_data_gt;
        canvas = VegaLite.@vlplot(width = 200, height = 200),
        legend = VegaLite.@vlfrag(orient = "top", offset = 5),
        trajectory_viz_domain...,
    ) + VegaLite.@vlplot(
        data = filter(s -> s.t == 1, trajectory_data_gt),
        mark = {"text", dx = 8, dy = 8},
        text = "player",
        x = "px",
        y = "py",
    )

observations_bundle_viz =
    VegaLite.@vlplot() + MonteCarloStudy.visualize_trajectory_batch(
        trajectory_data_obs;
        trajectory_viz_domain...,
        draw_line = false,
    )

viz_trajectory_estiamtes = Dict(
    k => TrajectoryVisualization.visualize_trajectory_batch(v; trajectory_viz_domain...) for
    (k, v) in trajectory_data_estimates
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

frame = [-1.5n_observation_sequences_per_noise_level, 0]
@saveviz parameter_error_viz =
    errstats |> MonteCarloStudy.visualize_paramerr(; frame, round_x_axis = false)
@saveviz position_error_viz =
    errstats |> MonteCarloStudy.visualize_poserr(; frame, round_x_axis = false)

==#
