struct RandomEstimator end

function estimate(::RandomEstimator; dataset, player_cost_models, kwargs...)
    rng = Random.MersenneTwister(1)

    map(eachindex(dataset)) do observation_idx
        player_weights_random = map(player_cost_models) do cost_model
            normalized_random_weights =
                LinearAlgebra.normalize(rand(rng, length(cost_model.weights)), 1)
            NamedTuple{keys(cost_model.weights)}(normalized_random_weights)
        end
        (; player_weights = player_weights_random, converged = true, observation_idx)
    end
end


estimates_random = estimate(RandomEstimator(); estimator_setup...)

augmented_estimates_random = run_cached!(:augmented_estimates_random) do
    augment_with_forward_solution(estimates_random; augmentor_kwargs...)
end

errstats_random =
    estimator_statistics(augmented_estimates_random, dataset; demo_gt, estimator_name = "Random")
