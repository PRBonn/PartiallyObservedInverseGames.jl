const project_root_dir = realpath(joinpath(@__DIR__, ".."))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "experiments/utils/MonteCarloStudy"))))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "test/utils"))))

import Distributed
Distributed.@everywhere begin
    import Pkg
    Pkg.activate($project_root_dir)
    union!(LOAD_PATH, $LOAD_PATH)

    import MonteCarloStudy
    import CollisionAvoidanceGame
    import TestDynamics
    using PartiallyObservedInverseGames.ForwardGame: IBRGameSolver, KKTGameSolver
    using PartiallyObservedInverseGames.InverseGames: InverseKKTConstraintSolver, InverseKKTResidualSolver

end

import PartiallyObservedInverseGames.TrajectoryVisualization
import ElectronDisplay

# Utils
include("utils/misc.jl")
include("utils/simple_caching.jl")
load_cache_if_not_defined!("unicycle")

#==================================== Forward Game Formulation =====================================#

T = 25

control_system = TestDynamics.ProductSystem([
    TestDynamics.Unicycle(0.25),
    TestDynamics.Unicycle(0.25),
    # TestDynamics.Unicycle(0.25),
])

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
demo_noise_level = 0.1

ground_truth_viz =
    TrajectoryVisualization.visualize_trajectory(
        control_system,
        forward_solution_gt.x,
        TrajectoryVisualization.VegaLiteBackend(),
        canvas = VegaLite.@vlplot(width = 200, height = 200),
        legend = VegaLite.@vlfrag(orient = "top", offset = 5),
    ) + VegaLite.@vlplot(
        data = [
            (;
                px = forward_solution_gt.x[TestDynamics.state_indices(control_system, ii)[1], 1],
                py = forward_solution_gt.x[TestDynamics.state_indices(control_system, ii)[2], 1],
                player = "P$ii",
            ) for ii in eachindex(control_system.subsystems)
        ],
        mark = {"text", dx = 8, dy = 8},
        text = "player",
        x = "px",
        y = "py",
    )

observations_bundle_viz =
    VegaLite.@vlplot() + MonteCarloStudy.visualize_trajectory_batch(
        control_system,
        [d.x for d in dataset if d.σ == demo_noise_level],
        x_position_domain = (-1.2, 1.2),
        y_position_domain = (-1.2, 1.2),
        draw_line = false,
    )

conKKT_bundle_viz = TrajectoryVisualization.visualize_trajectory_batch(
    control_system,
    [
        e.x
        for
        e in estimates_conKKT if e.converged && dataset[e.observation_idx].σ == demo_noise_level
    ];
    x_position_domain = (-1.2, 1.2),
    y_position_domain = (-1.2, 1.2),
)

conKKT_partial_bundle_viz = TrajectoryVisualization.visualize_trajectory_batch(
    control_system,
    [
        e.x
        for
        e in estimates_conKKT_partial if
        e.converged && dataset[e.observation_idx].σ == demo_noise_level
    ];
    x_position_domain = (-1.2, 1.2),
    y_position_domain = (-1.2, 1.2),
)

resKKT_bundle_viz = TrajectoryVisualization.visualize_trajectory_batch(
    control_system,
    [
        e.x
        for
        e in augmented_estimates_resKKT if
        e.converged && dataset[e.observation_idx].σ == demo_noise_level
    ];
    x_position_domain = (-1.2, 1.2),
    y_position_domain = (-1.2, 1.2),
)

resKKT_partial_bundle_viz = TrajectoryVisualization.visualize_trajectory_batch(
    control_system,
    [
        e.x
        for
        e in augmented_estimates_resKKT_partial if
        e.converged && dataset[e.observation_idx].σ == demo_noise_level
    ];
    x_position_domain = (-1.2, 1.2),
    y_position_domain = (-1.2, 1.2),
)

@saveviz demo_trajs_viz = hcat(
    VegaLite.@vlplot(title = "Ground Truth") + ground_truth_viz,
    VegaLite.@vlplot(title = "Observations") + observations_bundle_viz,
)

@saveviz ours_trajs_viz = hcat(
    VegaLite.@vlplot(title = "Full Observation") + conKKT_bundle_viz,
    VegaLite.@vlplot(title = "Partial Observation") + conKKT_partial_bundle_viz,
)

@saveviz baseline_trajs_viz = hcat(
    VegaLite.@vlplot(title = "Full Observation") + resKKT_bundle_viz,
    VegaLite.@vlplot(title = "Partial Observation") + resKKT_partial_bundle_viz,
)

frame = [-1.5n_observation_sequences_per_noise_level, 0]
@saveviz parameter_error_viz =
    errstats |> MonteCarloStudy.visualize_paramerr(; frame, round_x_axis = false)
@saveviz position_error_viz =
    errstats |> MonteCarloStudy.visualize_poserr(; frame, round_x_axis = false)
