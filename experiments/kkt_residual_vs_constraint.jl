unique!(push!(LOAD_PATH, realpath(joinpath(@__DIR__, "../test/utils"))))

import Random
import Statistics

import CollisionAvoidanceGame
import TestDynamics
import JuMPOptimalControl.CostUtils
using JuMPOptimalControl.DynamicsModelInterface: visualize_trajectory
using JuMPOptimalControl.ForwardGame: IBRGameSolver, solve_game
using JuMPOptimalControl.InverseGames:
    InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game

using ProgressMeter: @showprogress
using VegaLite: @vlplot

#=================================== Simple caching / memoization ==================================#

# NOTE: clear cache via `empty!(results_cache)`
if !isdefined(Main, :results_cache)
    results_cache = Dict()
end

function cached_get(f, cache_dict, key)
    result = get(f, cache_dict, key)
    cache_dict[key] = result
    result
end

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
# here.)
function generate_dataset(
    solve_args = (IBRGameSolver(), control_system, player_cost_models_gt, x0, T),
    solve_kwargs = (; solver_attributes = (; print_level = 1)),
    noise_levels = [0:0.002:0.01; 0.02:0.01:0.1 ;],
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

forward_solution_gt, dataset = cached_get(results_cache, :forward_solution_gt_dataset) do
    generate_dataset()
end

#========================================= Run estimators ==========================================#

# TODO: code duplication could be reduced if both solvers supported the same interface
function estimate(
    solver::InverseKKTConstraintSolver;
    dataset,
    control_system,
    player_cost_models,
    solver_attributes,
)

    @showprogress map(enumerate(dataset)) do (ii, d)
        observation_model = (; d.σ, expected_observation = identity)

        converged, estimate, opt_model = solve_inverse_game(
            solver,
            d.x;
            control_system,
            observation_model,
            player_cost_models,
            solver_attributes,
            # NOTE: right now this does not use the u information for initialization.
        )
        converged || @warn "conKKT did not converge on observation $ii."

        merge(estimate, (; converged))
    end
end

function estimate(
    solver::InverseKKTResidualSolver;
    dataset = dataset,
    control_system = control_system,
    player_cost_models = player_cost_models_gt,
    solver_attributes = (; print_level = 1),
)
    @showprogress map(enumerate(dataset)) do (ii, d)
        converged, estimate, opt_model = solve_inverse_game(
            solver,
            d.x,
            d.u;
            control_system,
            player_cost_models,
            solver_attributes,
        )
        converged || @warn "resKKT did not converge on observation $ii."

        merge(estimate, (; converged))
    end
end

estimator_setup = (;
    dataset,
    control_system,
    player_cost_models = player_cost_models_gt,
    solver_attributes = (; print_level = 1),
)

estimates_conKKT = cached_get(results_cache, :estimates_conKKT) do
    estimate(InverseKKTConstraintSolver(); estimator_setup...)
end

estimates_resKKT = cached_get(results_cache, :estimates_resKKT) do
    estimate(InverseKKTResidualSolver(); estimator_setup...)
end

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

        converged || @warn "Forward KKT did not converge on observation $ii."

        merge(estimate, (; forward_solution.x, forward_solution.u, converged))
    end
end

augmented_estimtes_resKKT = cached_get(results_cache, :augmented_estimtes_resKKT) do
    augment_with_forward_solution(
        estimates_resKKT;
        solver = IBRGameSolver(),
        control_system,
        player_cost_models_gt,
        x0,
        T,
        # Init with forward_solution_gt trajectory to make sure we are recoving the correct equilibrium
        init = (; forward_solution_gt.x, forward_solution_gt.u),
        solver_attributes = (; print_level = 1),
    )
end

#===================================== Statistical Evaluation ======================================#

function estimator_statistics(estimates::AbstractVector, observations::AbstractVector; kwargs...)
    map((args...) -> estimator_statistics(args...; kwargs...), estimates, observations)
end

function estimator_statistics(estimate, observation; demo_gt, estimator_name)
    mean_abs_err_x_obs = Statistics.mean(abs.(demo_gt.x - observation.x))
    mean_abs_err_u_obs = Statistics.mean(abs.(demo_gt.u - observation.u))
    mean_abs_err_pos_obs =
        Statistics.mean(abs.(demo_gt.x[position_indices, :] - observation.x[position_indices, :]))

    mean_abs_err_x_est = Statistics.mean(abs.(demo_gt.x - estimate.x))
    mean_abs_err_u_est = Statistics.mean(abs.(demo_gt.u - estimate.u))
    mean_abs_err_pos_est =
        Statistics.mean(abs.(demo_gt.x[position_indices, :] - estimate.x[position_indices, :]))

    mean_rel_weight_err =
        map(demo_gt.player_cost_models_gt, estimate.player_weights) do cost_model_gt, weights_est
            @assert sum(weights_est) ≈ 1
            map(CostUtils.normalize(cost_model_gt.weights), weights_est) do weight_gt, weight_est
                abs(weight_gt - weight_est) / weight_gt
            end |> Statistics.mean
        end |> Statistics.mean

    (;
        estimator_name,
        estimate.converged,
        mean_abs_err_x_obs,
        mean_abs_err_u_obs,
        mean_abs_err_pos_obs,
        mean_abs_err_x_est,
        mean_abs_err_u_est,
        mean_abs_err_pos_est,
        mean_rel_weight_err,
    )
end

demo_gt = merge((; player_cost_models_gt), forward_solution_gt)
errstats_conKKT =
    estimator_statistics(estimates_conKKT, dataset; demo_gt, estimator_name = "KKT Constraints")
errstats_resKKT = estimator_statistics(
    augmented_estimtes_resKKT,
    dataset;
    demo_gt,
    estimator_name = "KKT Residuals",
)

#========================================== Visualization ==========================================#

import ElectronDisplay

parameter_error_visualizer = @vlplot(
    mark = :point,
    x = {:mean_abs_err_x_obs, title = "Mean Absolute State Observation Noise"},
    y = {:mean_rel_weight_err, title = "Mean Relative Parameter Error"},
    color = {:estimator_name, title = "Estimator"},
    shape = {:estimator_name, title = "Estimator"},
    fill = {
        :converged,
        title = "Forward Solution Converged",
        type = "nominal",
        scale = {scheme = "set1"},
    },
    width = 700,
    height = 400,
)

position_error_visualizer = @vlplot(
    mark = {:point, size = 75},
    x = {:mean_abs_err_pos_obs, title = "Mean Absolute Postion Observation Error [m]"},
    y = {:mean_abs_err_pos_est, title = "Mean Absolute Position Prediction Error [m]"},
    color = {:estimator_name, title = "Estimator"},
    shape = {:estimator_name, title = "Estimator"},
    fill = {
        :converged,
        title = "Forward Solution Converged",
        type = "nominal",
        scale = {scheme = "set1"},
    },
    width = 700,
    height = 400,
)

errstats = [errstats_conKKT; errstats_resKKT]

errstats |> @vlplot() + [position_error_visualizer; parameter_error_visualizer]
