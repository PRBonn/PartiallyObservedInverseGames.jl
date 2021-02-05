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
T = 25
player_cost_models = let
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
#
# TODO it would be nice if both solver had the same interface (i.e. the `InverseKKTConstraintSolver`
# should allow to take an `u` as second argument. Maybe add a distpatch for that.)

if recreate_dataset || !isdefined(Main, :dataset)
    forward_solution_gt, forward_opt_model_gt =
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T;)
    # TODO check that the forward thing converged.
    dataset = generate_observations(forward_solution_gt.x, forward_solution_gt.u)
end

if rerun_experiments || !isdefined(Main, :estimates_conKKT) || !isdefined(Main, :estimates_resKKT)
    estimates_conKKT = @showprogress map(dataset) do d
        observation_model = (; d.σ, expected_observation = identity)

        estimate, opt_model = solve_inverse_game(
            InverseKKTConstraintSolver(),
            d.x;
            control_system,
            observation_model,
            player_cost_models,
            solver_attributes = (; print_level = 1),
            # NOTE: right now this does not use the u information for initialization.
        )

        estimate
    end

    estimates_resKKT = @showprogress map(dataset) do d
        observation_model = (; d.σ, expected_observation = identity)

        estimate, opt_model = solve_inverse_game(
            InverseKKTResidualSolver(),
            d.x,
            d.u;
            control_system,
            player_cost_models,
            solver_attributes = (; print_level = 1),
        )

        estimate
    end
end

#===================================== Statistical Evaluation ======================================#

function error_statistics(ground_truth, observation, estimate)
    x_observation_error = (ground_truth.x - observation.x)
    u_observation_error = (ground_truth.u - observation.u)

    player_weight_relerrors =
        map(ground_truth.player_cost_models, estimate.player_weights) do cost_model_gt, weights_est
            @assert sum(weights_est) ≈ 1
            map(
                (weight_gt, weight_est) -> abs(weight_gt - weight_est) / weight_gt,
                CostUtils.normalize(cost_model_gt.weights),
                CostUtils.namedtuple(weights_est),
            )
        end

    # TODO: is this the right metric?
    x_estimate_error =
        hasproperty(estimate, :x) ? (ground_truth.x - estimate.x) : x_observation_error
    u_estimate_error =
        hasproperty(estimate, :u) ? (ground_truth.x - estimate.x) : u_observation_error

    (;
        x_observation_error,
        u_observation_error,
        player_weight_relerrors,
        x_estimate_error,
        u_estimate_error,
    )
end

# TODO: continue here
ground_truth = merge((; player_cost_models), forward_solution_gt)
errstats_conKKT = map(estimates_conKKT, dataset) do estimate, observation
    error_statistics(ground_truth, observation, estimate)
end
errstats_resKKT = map(estimates_resKKT, dataset) do estimate, observation
    error_statistics(ground_truth, observation, estimate)
end

#========================================== Visualization ==========================================#

using Query: @map
import ElectronDisplay

# TODO: may do not average over different state, input, and parameter dimension because that messes
# with the units. But I'm not sure what would be a good projection then (we can't show all
# combinations of dimensions. There are too many).
summarize_stats = @map({
    mean_abs_err_x_obs = Statistics.mean(abs.(_.x_observation_error)),
    mean_abs_err_u_obs = Statistics.mean(abs.(_.u_observation_error)),
    mean_abs_err_x_est = Statistics.mean(abs.(_.x_estimate_error)),
    mean_abs_err_u_est = Statistics.mean(abs.(_.u_estimate_error)),
    mean_rel_weight_err = Statistics.mean(Statistics.mean.(_.player_weight_relerrors)),
})

parameter_error_visualizer = @vlplot(
    mark = :point,
    x = {:mean_abs_err_x_obs, title = "Mean Absolute State Observation Noise"},
    y = {:mean_rel_weight_err, title = "Mean Relative Parameter Error"},
    color = :Estimator,
    shape = :Estimator,
    width = 700,
    height = 400
)

errstats = vcat(
    errstats_conKKT |> summarize_stats |> @map({_..., Estimator = "KKT Constraints"}) |> collect,
    errstats_resKKT |> summarize_stats |> @map({_..., Estimator = "KKT Residuals"}) |> collect,
) |> parameter_error_visualizer

# TODO: Augment the KKT residual estimates with solutions (essentially their x estimate) and
# visualize the error.

# if !isdefined(Main, :forward_solutions_resKKT) || recreate_forward_solutions
#     forward_solutions_resKKT = @showprogress map(estimates_resKKT) do estimate
#         forward_solution, forward_opt_model = solve_game(
#             KKTGameSolver(),
#             control_system,
#             map(
#                 (cost_model_gt, weights) -> merge(cost_model_gt, (; weights)),
#                 player_cost_models,
#                 estimate.player_weights,
#             ),
#             x0,
#             T;
#             init = forward_solution_gt,
#             solver_attributes = (; print_level = 3),
#         )
#         # TODO: check that the solver converged
# 
#         forward_solution
#     end
# end
# 
# forward_solutions_conKKT = estimates_conKKT
# 
# function trajectory_error_stats(solutions, dataset, estimator_name)
#     map(solutions, dataset) do solution, demo
# 
#         function trajectory_error_stats(x_gt, x_sol)
#             Statistics.mean(abs.(x_gt[1:2, :] - x_sol[1:2, :]))
#         end
# 
#         mean_abs_err_x_sol = trajectory_error_stats(forward_solution_gt.x, solution.x)
#         mean_abs_err_x_obs = trajectory_error_stats(forward_solution_gt.x, demo.x)
# 
#         (; mean_abs_err_x_sol, mean_abs_err_x_obs, estimator_name = estimator_name)
#     end
# end
# 
# vcat(
#     trajectory_error_stats(forward_solutions_conKKT, dataset, "KKT Constraints"),
#     trajectory_error_stats(forward_solutions_resKKT, dataset, "KKT Residuals"),
# ) |> @vlplot(
#     mark = :point,
#     x = :mean_abs_err_x_obs,
#     y = :mean_abs_err_x_sol,
#     color = :estimator_name,
#     shape = :estimator_name,
#     width = 700,
#     height = 400
# )
# 
