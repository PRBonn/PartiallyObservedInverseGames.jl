function estimator_statistics(
    estimate;
    dataset,
    player_cost_models_gt,
    position_indices,
    trajectory_distance = Distances.meanad,
    parameter_distance = Distances.cosine_dist,
)
    function trajectory_component_errors(t1, t2; window = 1:size(t2.x, 2))
        if haskey(t2, :x)
            (;
                x_error = trajectory_distance(t1.x[:, window], t2.x[:, window]),
                position_error = trajectory_distance(
                    t1.x[position_indices, window],
                    t2.x[position_indices, window],
                ),
            )
        else
            (; x_error = Inf, position_error = Inf)
        end
    end

    d = dataset[estimate.observation_idx]
    x_observation_error, position_observation_error =
        trajectory_component_errors(d.ground_truth, d.observation)
    x_estimation_error, position_estimation_error =
        trajectory_component_errors(d.ground_truth, estimate)

    parameter_estimation_error =
        map(player_cost_models_gt, estimate.player_weights) do cost_model_gt, weights_est
            @assert sum(weights_est) ≈ 1
            parameter_distance(CostUtils.normalize(cost_model_gt.weights), weights_est)
        end |> Statistics.mean

    (;
        estimate.estimator_name,
        estimate.observation_idx,
        estimate.converged,
        d.σ,
        d.observation_horizon,
        x_observation_error,
        position_observation_error,
        x_estimation_error,
        position_estimation_error,
        parameter_estimation_error,
    )
end
