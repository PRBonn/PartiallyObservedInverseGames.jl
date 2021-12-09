function estimator_statistics(
    estimate;
    dataset,
    demo_gt,
    position_indices,
    trajectory_distance = Distances.meanad,
    parameter_distance = Distances.cosine_dist,
)
    function trajectory_component_errors(trajectory)
        window = 1:size(trajectory.x, 2)
        if haskey(trajectory, :x)
            (;
                x_error = trajectory_distance(demo_gt.x[:, window], trajectory.x[:, window]),
                position_error = trajectory_distance(
                    demo_gt.x[position_indices, window],
                    trajectory.x[position_indices, window],
                ),
            )
        else
            (; x_error = Inf, position_error = Inf)
        end
    end

    observation = dataset[estimate.observation_idx]
    x_observation_error, position_observation_error = trajectory_component_errors(observation)
    x_estimation_error, position_estimation_error = trajectory_component_errors(estimate)

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
        position_observation_error,
        x_estimation_error,
        position_estimation_error,
        parameter_estimation_error,
    )
end
