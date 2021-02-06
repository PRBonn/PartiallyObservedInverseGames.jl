unique!(push!(LOAD_PATH, realpath(joinpath(@__DIR__, "../test/utils"))))

import Random
import Statistics
import Query

import CollisionAvoidanceGame
import TestDynamics
import JuMPOptimalControl.CostUtils
using JuMPOptimalControl.DynamicsModelInterface: visualize_trajectory
using JuMPOptimalControl.ForwardGame: KKTGameSolver, solve_game
using JuMPOptimalControl.InverseGames:
    InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game

using ProgressMeter: @showprogress
using VegaLite: @vlplot

#========================================== Configuration ==========================================#

recreate_dataset = false
rerun_experiments = false
recreate_forward_solutions = false

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
function generate_observations(
    x,
    u;
    noise_levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    n_trajectory_samples_per_noise_level = 10,
    rng = Random.MersenneTwister(1),
)

    # Note: reducing over inner loop to flatten the dataset
    mapreduce(vcat, noise_levels) do σ
        n_samples = iszero(σ) ? 1 : n_trajectory_samples_per_noise_level
        observations = map(1:n_samples) do _
            (; σ, x = x + σ * randn(rng, size(x)), u = u + σ * randn(rng, size(u)))
        end
    end
end

# TODO: Save the results data using something like JLD or BSON.
# TODO: Run experiments in parallel and/or distributed fashion (Not sure whether we can have more
# than one process talking to IPOPT on the same machine.)
# TODO: For now consider only one true forward game. Later also consider varying:
# - cost parameters of the observed system
# - initial conditions
# - initialization of forward solver (i.e. different equilibria)
# In this case we need to think about how to visualize the data in a meaningful way.

if recreate_dataset || !isdefined(Main, :dataset)
    converged, forward_solution_gt, forward_opt_model_gt =
        solve_game(KKTGameSolver(), control_system, player_cost_models_gt, x0, T;)
    @assert converged
    dataset = generate_observations(forward_solution_gt.x, forward_solution_gt.u)
end

if rerun_experiments || !isdefined(Main, :estimates_conKKT) || !isdefined(Main, :estimates_resKKT)
    estimates_conKKT = @showprogress map(dataset) do d
        observation_model = (; d.σ, expected_observation = identity)

        converged, estimate, opt_model = solve_inverse_game(
            InverseKKTConstraintSolver(),
            d.x;
            control_system,
            observation_model,
            player_cost_models_gt,
            solver_attributes = (; print_level = 1),
            # NOTE: right now this does not use the u information for initialization.
        )
        @assert converged

        estimate
    end

    estimates_resKKT = @showprogress map(dataset) do d
        converged, estimate, opt_model = solve_inverse_game(
            InverseKKTResidualSolver(),
            d.x,
            d.u;
            control_system,
            player_cost_models_gt,
            solver_attributes = (; print_level = 1),
        )
        @assert converged

        estimate
    end
end

#======== Augment KKT Residual Solution with State and Input Estimate via Forward Solution =========#

if recreate_forward_solutions || !isdefined(Main, :augmented_estimtes_resKKT)
    augmented_estimtes_resKKT = map(estimates_resKKT) do estimate
        # overwrite the weights of the ground truth model with the weights of the estimate.
        player_cost_models_est =
            map(player_cost_models_gt, estimate.player_weights) do cost_model_gt, weights
                merge(cost_model_gt, (; weights))
            end
        # solve the forward game at this point
        converged, forward_solution, forward_opt_model = solve_game(
            KKTGameSolver(),
            control_system,
            player_cost_models_est,
            x0,
            T;
            # Init with forward_solution_gt trajectory to make sure we are recoving the correct
            # equilequilibrium
            init = (; forward_solution_gt.x, forward_solution_gt.u),
        )
        @assert converged
        merge(estimate, (; forward_solution.x, forward_solution.u))
    end
end

#===================================== Statistical Evaluation ======================================#

# TODO: We may not want to average over different state, input, and parameter dimension because that
# messes with the units. But I'm not sure what would be a good projection then (we can't show all
# combinations of dimensions. There are too many).

function error_statistics(estimates::AbstractVector, observations::AbstractVector; kwargs...)
    map((args...) -> error_statistics(args...; kwargs...), estimates, observations)
end

function error_statistics(estimate, observation; demo_gt, estimator_name)
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
            map(
                CostUtils.normalize(cost_model_gt.weights),
                weights_est,
            ) do weight_gt, weight_est
                abs(weight_gt - weight_est) / weight_gt
            end |> Statistics.mean
        end |> Statistics.mean

    (;
        estimator_name,
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
    error_statistics(estimates_conKKT, dataset; demo_gt, estimator_name = "KKT Constraints")
errstats_resKKT =
    error_statistics(augmented_estimtes_resKKT, dataset; demo_gt, estimator_name = "KKT Residuals")

#========================================== Visualization ==========================================#

import ElectronDisplay

parameter_error_visualizer = @vlplot(
    mark = :point,
    x = {:mean_abs_err_x_obs, title = "Mean Absolute State Observation Noise"},
    y = {:mean_rel_weight_err, title = "Mean Relative Parameter Error"},
    color = {:estimator_name, title = "Estimator"},
    shape = {:estimator_name},
    width = 700,
    height = 400,
)

position_error_visualizer = @vlplot(
    mark = :point,
    x = {:mean_abs_err_pos_obs, title = "Mean Absolute Postion Observation Error [m]"},
    y = {:mean_abs_err_pos_est, title = "Mean Absolute Position Prediction Error [m]"},
    color = {:estimator_name, title = "Estimator"},
    shape = {:estimator_name},
    width = 700,
    height = 400,
)

errstats = [errstats_conKKT; errstats_resKKT]
errstats |> position_error_visualizer
