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
import JuMPOptimalControl.CostUtils
import JuMPOptimalControl.InversePreSolve
using JuMPOptimalControl.TrajectoryVisualization: visualize_trajectory, visualize_trajectory_batch
using JuMPOptimalControl.ForwardGame: KKTGameSolver, IBRGameSolver, solve_game
using JuMPOptimalControl.InverseGames:
    InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game

using ProgressMeter: @showprogress
using VegaLite: @vlplot

# Utils
include("./simple_caching.jl")
include("./utils.jl")

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

#======================================== Generate Dataset =========================================#

# TODO: maybe allow for different noise levels per dimension (i.e. allow to pass covariance matrix
# here.). But in that case, we would also need to weigh the different dimensions with the correct
# information matrix in the KKT constraint approach.

function generate_dataset(
    solve_args = (IBRGameSolver(), control_system, player_cost_models_gt, x0, T),
    solve_kwargs = (; solver_attributes = (; print_level = 1)),
    noise_levels = unique([0:0.002:0.01; 0.02:0.01:0.1]),
    n_trajectory_samples_per_noise_level = 10,
    rng = Random.MersenneTwister(1),
)
    converged_gt, forward_solution_gt, forward_opt_model_gt =
        solve_game(solve_args...; solve_kwargs...)
    @assert converged_gt

    # Note: reducing over inner loop to flatten the dataset
    dataset = mapreduce(vcat, noise_levels) do σ
        n_samples = iszero(σ) ? 1 : n_trajectory_samples_per_noise_level
        observations = map(1:n_samples) do _
            (;
                σ,
                x = forward_solution_gt.x + σ * randn(rng, size(forward_solution_gt.x)),
                u = forward_solution_gt.u + σ * randn(rng, size(forward_solution_gt.u)),
            )
        end
    end

    forward_solution_gt, dataset
end

@run_cached forward_solution_gt, dataset = generate_dataset()

#========================================= Run estimators ==========================================#

# TODO: code duplication could be reduced if both solvers supported the same interface
function estimate(
    solver::InverseKKTConstraintSolver;
    dataset,
    control_system,
    player_cost_models,
    solver_attributes,
    expected_observation = identity,
    estimator_name = (expected_observation === identity ? "Ours Full" : "Ours Partial"),
    solver_kwargs...,
)

    @showprogress map(enumerate(dataset)) do (observation_idx, d)
        observation_model = (; d.σ, expected_observation)

        converged, estimate, opt_model = solve_inverse_game(
            solver,
            expected_observation(d.x);
            control_system,
            observation_model,
            player_cost_models,
            solver_attributes,
            solver_kwargs...,
            # NOTE: This estimator does not use any information beyond the state observation!
        )
        converged || @warn "conKKT did not converge on observation $observation_idx."

        merge(estimate, (; converged, observation_idx, estimator_name))
    end
end

function estimate(
    solver::InverseKKTResidualSolver;
    dataset = dataset,
    control_system = control_system,
    player_cost_models = player_cost_models_gt,
    solver_attributes = (; print_level = 1),
    expected_observation = identity,
    pre_solve = true,
    estimator_name = (expected_observation === identity ? "Baseline Full" : "Baseline Partial"),
)
    @showprogress map(enumerate(dataset)) do (observation_idx, d)
        observation_model = (; d.σ, expected_observation)
        observation = if pre_solve
            pre_solve_converged, pre_solve_solution = InversePreSolve.pre_solve(
                expected_observation(d.x),
                # TODO: think about  this. Should "fully observed" imply "observe inputs"?
                (expected_observation === identity ? d.u : nothing);
                control_system,
                observation_model,
                solver_attributes,
            )
            @assert pre_solve_converged
            (; pre_solve_solution.x, pre_solve_solution.u)
        else
            (; d.x, d.u)
        end

        converged, estimate, opt_model = solve_inverse_game(
            solver,
            observation.x,
            observation.u;
            control_system,
            player_cost_models,
            solver_attributes,
        )
        converged || @warn "resKKT did not converge on observation $observation_idx."

        merge(estimate, (; converged, observation_idx, estimator_name))
    end
end

estimator_setup = (;
    dataset,
    control_system,
    player_cost_models = player_cost_models_gt,
    solver_attributes = (; print_level = 1),
    pre_solve = true,
)

estimator_setup_partial =
    merge(estimator_setup, (; expected_observation = x -> x[partial_state_indices, :]))

@run_cached estimates_conKKT = estimate(InverseKKTConstraintSolver(); estimator_setup...)

@run_cached estimates_conKKT_partial =
    estimate(InverseKKTConstraintSolver(); estimator_setup_partial...)

@run_cached estimates_resKKT = estimate(InverseKKTResidualSolver(); estimator_setup...)

@run_cached estimates_resKKT_partial =
    estimate(InverseKKTResidualSolver(); estimator_setup_partial...)

#======== Augment KKT Residual Solution with State and Input Estimate via Forward Solution =========#

# TODO: this couuld also use the pre-solver to accelerate augmentation
function augment_with_forward_solution(
    estimates;
    solver,
    control_system,
    player_cost_models_gt,
    x0,
    T,
    kwargs...,
)
    @showprogress map(enumerate(estimates)) do (ii, estimate)
        # overwrite the weights of the ground truth model with the weights of the estimate.
        player_cost_models_est =
            map(player_cost_models_gt, estimate.player_weights) do cost_model_gt, weights
                merge(cost_model_gt, (; weights))
            end

        # solve the forward game at this point
        converged, forward_solution, forward_opt_model =
            solve_game(solver, control_system, player_cost_models_est, x0, T; kwargs...)

        converged || @warn "Forward solution augmentation did not converge on observation $ii."

        merge(estimate, (; forward_solution.x, forward_solution.u, converged))
    end
end

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
    augment_with_forward_solution(estimates_resKKT; augmentor_kwargs...)

@run_cached augmented_estimates_resKKT_partial =
    augment_with_forward_solution(estimates_resKKT_partial; augmentor_kwargs...)

estimates = [
    estimates_conKKT
    estimates_conKKT_partial
    augmented_estimates_resKKT
    augmented_estimates_resKKT_partial
]

#===================================== Statistical Evaluation ======================================#

function estimator_statistics(
    estimate;
    dataset,
    demo_gt,
    trajectory_distance = Distances.meanad,
    parameter_distance = Distances.cosine_dist,
)

    function trajectory_component_errors(trajectory)
        (;
            x_error = trajectory_distance(demo_gt.x, trajectory.x),
            u_error = trajectory_distance(demo_gt.u, trajectory.u),
            position_error = trajectory_distance(
                demo_gt.x[position_indices, :],
                trajectory.x[position_indices, :],
            ),
        )
    end

    observation = dataset[estimate.observation_idx]
    x_observation_error, u_observation_error, position_observation_error =
        trajectory_component_errors(observation)
    x_estimation_error, u_estimation_error, position_estimation_error =
        trajectory_component_errors(estimate)

    parameter_estimation_error =
        map(demo_gt.player_cost_models_gt, estimate.player_weights) do cost_model_gt, weights_est
            @assert sum(weights_est) ≈ 1
            parameter_distance(CostUtils.normalize(cost_model_gt.weights), weights_est)
        end |> Statistics.mean

    (;
        estimate.estimator_name,
        estimate.observation_idx,
        estimate.converged,
        observation.σ,
        x_observation_error,
        u_observation_error,
        position_observation_error,
        x_estimation_error,
        u_estimation_error,
        position_estimation_error,
        parameter_estimation_error,
    )
end

demo_gt = merge((; player_cost_models_gt), forward_solution_gt)

errstats = map(estimates) do estimate
    estimator_statistics(estimate; dataset, demo_gt)
end

#========================================== Visualization ==========================================#

"Visualize all trajectory `estimates` along with the corresponding ground truth
`forward_solution_gt`"
function visualize_bundle(
    control_system,
    estimates,
    forward_solution_gt;
    filter_converged = false,
    kwargs...,
)
    position_domain = extrema(forward_solution_gt.x[1:2, :]) .+ (-0.01, 0.01)
    estimated_trajectory_batch = [e.x for e in estimates if !filter_converged || e.converged]
    visualize_trajectory_batch(
        control_system,
        estimated_trajectory_batch;
        position_domain,
        kwargs...,
    )
end

global_config = VegaLite.@vlfrag(legend = {orient = "top", padding = 0})

function visualize_paramerr(;
    scatter_opacity = 0.2,
    width = 500,
    height = 300,
    y_label = "Mean Parameter Cosine Error",
)
    common_viz = @vlplot(
        color = {"estimator_name:n", title = "Estimator"},
        width = width,
        height = height,
        config = global_config
    )

    raw_scatter_viz = @vlplot(
        mark =
            {"point", tooltip = {content = "data"}, opacity = scatter_opacity, filled = true},
        x = {"position_observation_error:q", title = "Mean Absolute Postion Observation Error [m]"},
        y = {"parameter_estimation_error:q", title = y_label},
    )

    median_iqr_viz =
        @vlplot(
            x = {"σ:q"},
            y = {"parameter_estimation_error:q", aggregate = "median", title = y_label}
        ) +
        @vlplot(mark = "line") +
        @vlplot(mark = {"errorband", extent = "iqr"})

    common_viz + median_iqr_viz + raw_scatter_viz
end

function visualize_poserr(;
    scatter_opacity = 0.2,
    width = 500,
    height = 300,
    y_label = "Mean Absolute Position Prediciton Error [m]",
)
    common_viz = @vlplot(
        color = {"estimator_name:n", title = "Estimator"},
        width = width,
        height = height,
        config = global_config
    )

    raw_scatter_viz = @vlplot(
        mark =
            {"point", tooltip = {content = "data"}, opacity = scatter_opacity, filled = true},
        shape = {
            "converged:n",
            title = "Trajectory Reconstructable",
            legend = nothing,
            scale = {domain = [true, false], range = ["circle", "triangle-down"]},
        },
        x = {"position_observation_error:q", title = "Mean Absolute Postion Observation Error [m]"},
        y = {
            "position_estimation_error:q",
            title = y_label,
            scale = {type = "symlog", constant = 0.01},
        },
    )

    median_iqr_viz =
        @vlplot(
            x = {"σ:q"},
            y = {"position_estimation_error:q", aggregate = "median", title = y_label}
        ) +
        @vlplot(mark = "line") +
        @vlplot(mark = {"errorband", extent = "iqr"})

    common_viz + median_iqr_viz + raw_scatter_viz
end

@saveviz conKKT_partial_bundle_viz =
    visualize_bundle(control_system, estimates_conKKT_partial, forward_solution_gt)
@saveviz resKKT_partial_bundle_viz = visualize_bundle(
    control_system,
    augmented_estimates_resKKT_partial,
    forward_solution_gt;
    filter_converged = true,
)
@saveviz dataset_bundle_viz = visualize_bundle(control_system, dataset, forward_solution_gt)

@saveviz parameter_error_viz = errstats |> visualize_paramerr()
@saveviz position_error_viz = errstats |> visualize_poserr()
