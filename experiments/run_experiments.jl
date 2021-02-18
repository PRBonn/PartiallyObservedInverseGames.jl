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

include("./simple_caching.jl")
include("./saveviz.jl")

#==================================== Forward Game Formulation =====================================#

control_system =
    TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), TestDynamics.Unicycle(0.25)])

x0 = vcat([-1, 0, 0.1, 0 + deg2rad(10)], [0, -1, 0.1, pi / 2 + deg2rad(10)])
position_indices = [1, 2, 5, 6] # TODO: for now just hard-coded
T = 25

player_cost_models_gt = let
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model(;
        T,
        state_indices = 1:4,
        input_indices = 1:2,
        goal_position = [1, 0],
    )
    cost_model_p2 = CollisionAvoidanceGame.generate_player_cost_model(;
        T,
        state_indices = 5:8,
        input_indices = 3:4,
        goal_position = [0, 1],
    )

    (cost_model_p1, cost_model_p2)
end

#======================================== Generate Dataset =========================================#

# TODO: maybe allow for different noise levels per dimension (i.e. allow to pass covariance matrix
# here.). But in that case, we would also need to weigh the different dimensions with the correct
# information matrix in the KKT constraint approach.

function generate_dataset(
    solve_args = (IBRGameSolver(), control_system, player_cost_models_gt, x0, T),
    solve_kwargs = (; solver_attributes = (; print_level = 1)),
    noise_levels = [0:0.002:0.015; 0.02:0.01:0.1],
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
    estimator_name = (
        expected_observation === identity ? "KKT Constraints Full" : "KKT Constraints Partial"
    ),
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
    estimator_name = (
        expected_observation === identity ? "KKT Residuals Full" : "KKT Residuals Partial"
    ),
)
    @showprogress map(enumerate(dataset)) do (observation_idx, d)
        observation_model = (; d.σ, expected_observation)
        observation = if pre_solve
            pre_solve_converged, pre_solve_solution = InversePreSolve.pre_solve(
                expected_observation(d.x),
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
    merge(estimator_setup, (; expected_observation = x -> x[[1, 2, 4, 5, 6, 8], :]))

@run_cached estimates_conKKT = estimate(InverseKKTConstraintSolver(); estimator_setup...)

@run_cached estimates_conKKT_partial =
    estimate(InverseKKTConstraintSolver(); estimator_setup_partial...)

@run_cached estimates_resKKT = estimate(InverseKKTResidualSolver(); estimator_setup...)

@run_cached estimates_resKKT_partial =
    estimate(InverseKKTResidualSolver(); estimator_setup_partial...)

#======== Augment KKT Residual Solution with State and Input Estimate via Forward Solution =========#

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

# TODO: add the estimator name already earlier (when estimtes are created).
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

# TODO: share more of the spec to avoid duplication.

parameter_error_visualizer = @vlplot(
    mark = {:point, size = 50, tooltip = {content = "data"}},
    x = {:position_observation_error, title = "Mean Absolute Postion Observation Error [m]"},
    y = {:parameter_estimation_error, title = "Mean Parameter Cosine Error"},
    color = {:estimator_name, title = "Estimator"},
    shape = {:estimator_name, title = "Estimator"},
    width = 500,
    height = 300,
    config = global_config,
    #    title = {text = "(a) Position Error", fontWeight = "normal", orient = "bottom"}jjjjj
)

position_error_visualizer = @vlplot(
    mark = {:point, size = 50, tooltip = {content = "data"}},
    x = {:position_observation_error, title = "Mean Absolute Postion Observation Error [m]"},
    y = {
        :position_estimation_error,
        scale = {type = "symlog", constant = 0.01},
        title = "Mean Absolute Position Prediction Error [m]",
    },
    color = {:estimator_name, title = "Estimator", legend = nothing},
    shape = {:estimator_name, title = "Estimator", legend = nothing},
    fill = {
        :converged,
        title = "Trajectory Reconstructable",
        type = "nominal",
        scale = {domain = [true, false], range = ["bisque", "tomato"]},
        legend = nothing,
    },
    width = 500,
    height = 300,
    config = global_config
)

@saveviz conKKT_partial_bundle_viz =
    visualize_bundle(control_system, estimates_conKKT_partial, forward_solution_gt)
@saveviz resKKT_partial_bundle_viz = visualize_bundle(
    control_system,
    augmented_estimates_resKKT_partial,
    forward_solution_gt;
    filter_converged = true,
)

@saveviz position_error_viz = errstats |> position_error_visualizer
@saveviz parameter_error_viz = errstats |> parameter_error_visualizer
@saveviz dataset_bundle_viz = visualize_bundle(control_system, dataset, forward_solution_gt)
