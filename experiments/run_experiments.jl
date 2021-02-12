unique!(push!(LOAD_PATH, realpath(joinpath(@__DIR__, "../test/utils"))))

import Random
import Statistics
import LinearAlgebra
import Distances

import CollisionAvoidanceGame
import TestDynamics
import JuMPOptimalControl.CostUtils
using JuMPOptimalControl.TrajectoryVisualization: visualize_trajectory, visualize_trajectory_batch
using JuMPOptimalControl.ForwardGame: KKTGameSolver, IBRGameSolver, solve_game
using JuMPOptimalControl.InverseGames:
    InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game

using ProgressMeter: @showprogress
using VegaLite: @vlplot

include("./simple_caching.jl")

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
        expected_observation === identity ? "KKT Constraints" : "KKT Constraints Partial"
    ),
    solver_kwargs...
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
            solver_kwargs...
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
    estimator_name = "KKT Residuals",
)
    @showprogress map(enumerate(dataset)) do (observation_idx, d)
        converged, estimate, opt_model = solve_inverse_game(
            solver,
            d.x,
            d.u;
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
)

@run_cached estimates_conKKT = estimate(InverseKKTConstraintSolver(); estimator_setup...)

@run_cached estimates_conKKT_partial = estimate(
    InverseKKTConstraintSolver();
    estimator_setup...,
    expected_observation = x -> x[[1, 2, 4, 5, 6, 8], :],
    pre_solve = true,
)

@run_cached estimates_resKKT = estimate(InverseKKTResidualSolver(); estimator_setup...)

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

@run_cached augmented_estimates_conKKT_partial =
    augment_with_forward_solution(estimates_conKKT_partial; augmentor_kwargs...)

@run_cached augmented_estimates_resKKT =
    augment_with_forward_solution(estimates_resKKT; augmentor_kwargs...)

estimates = [estimates_conKKT; estimates_conKKT_partial; augmented_estimates_resKKT]

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
function visualize_estimates(control_system, estimates, forward_solution_gt; only_converged = true)
    position_domain = extrema(forward_solution_gt.x[1:2, :]) .+ (-0.01, 0.01)
    estimated_trajectory_batch = [e.x for e in estimates if !only_converged || e.converged]
    visualize_trajectory_batch(control_system, estimated_trajectory_batch; position_domain)
end

import ElectronDisplay

# TODO: share more of the spec to avoid duplication.
position_error_visualizer = @vlplot(
    mark = {:point, size = 75, tooltip = {content = "data"}},
    x = {:position_observation_error, title = "Mean Absolute Postion Observation Error [m]"},
    y = {
        :position_estimation_error,
        scale = {type = "symlog", constant = 0.01},
        title = "Mean Absolute Position Prediction Error [m]",
    },
    color = {:estimator_name, title = "Estimator"},
    shape = {:estimator_name, title = "Estimator"},
    fill = {
        :converged,
        title = "Trajectory Reconstructable",
        type = "nominal",
        scale = {domain = [true, false], range = ["bisque", "tomato"]},
    },
    width = 700,
    height = 400,
)

parameter_error_visualizer = @vlplot(
    mark = {:point, size = 75, tooltip = {content = "data"}},
    x = {:position_observation_error, title = "Mean Absolute Postion Observation Error [m]"},
    y = {:parameter_estimation_error, title = "Mean Parameter Cosine Error"},
    color = {:estimator_name, title = "Estimator"},
    shape = {:estimator_name, title = "Estimator"},
    width = 700,
    height = 400,
)

errstats_visualization =
    errstats |> @vlplot() + [
        position_error_visualizer
        parameter_error_visualizer
    ]
