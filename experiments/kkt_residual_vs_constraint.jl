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
    noise_levels = [0, 0.01, 0.05, 0.1],
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
    forward_solution, forward_opt_model =
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T;)

    dataset = generate_observations(forward_solution.x, forward_solution.u)
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

    player_weight_errors =
        map(ground_truth.player_cost_models, estimate.player_weights) do cost_model_gt, weights_est
            # @assert sum(weights_est) ≈ 1
            map(
                (gt, est) -> abs(gt - est),
                CostUtils.normalize(cost_model_gt.weights),
                CostUtils.namedtuple(weights_est),
            )
        end

    # TODO: is this the right metric?
    x_estimate_error =
        hasproperty(estimate, :x) ? (ground_truth.x - estimate.x) :
        x_observation_error
    u_estimate_error =
        hasproperty(estimate, :u) ? (ground_truth.x - estimate.x) :
        u_observation_error

    (;
        x_observation_error,
        u_observation_error,
        player_weight_errors,
        x_estimate_error,
        u_estimate_error,
    )
end

# TODO: continue here
ground_truth = merge((; player_cost_models), forward_solution)
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
# combinations of dimensions. Threre are too many).
# TODO: For scale it may be easier to look at the standard deviation rather than the variance
summarize_stats = @map({
    σ_x_obs = Statistics.mean(abs.(_.x_observation_error)),
    σ_u_obs = Statistics.mean(abs.(_.u_observation_error)),
    σ_x_est = Statistics.mean(abs.(_.x_estimate_error)),
    σ_u_est = Statistics.mean(abs.(_.u_estimate_error)),
    mean_abs_σ_err = Statistics.mean(Statistics.mean.(_.player_weight_errors)),
})

vcat(
    errstats_resKKT |> summarize_stats |> @map({_..., Estimator = "KKT Residuals"}) |> collect,
    errstats_conKKT |> summarize_stats |> @map({_..., Estimator = "KKT Constraints"}) |> collect,
) |> @vlplot(
    mark = :point,
    x = {:σ_x_obs, title = "Mean Absolute State Observation Noise"},
    y = {:mean_abs_σ_err, title = "Mean Absolute Parameter Error"},
    color = :Estimator,
    shape = :Estimator,
    width = 700,
    height = 400
)
