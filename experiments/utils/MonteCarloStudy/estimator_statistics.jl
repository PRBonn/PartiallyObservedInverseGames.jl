function estimator_statistics(
    sample;
    player_cost_models_gt,
    position_indices,
    trajectory_distance = Distances.meanad,
    parameter_distance = Distances.cosine_dist,
    window_type,
    T_predict,
)
    T_obs = size(sample.observation.x, 2)
    window = let
        if window_type === :observation
            1:T_obs
        elseif window_type === :prediction
            (T_obs + 1):(T_obs + T_predict)
        end
    end

    # TODO: proper window sizing; I guess it's correct to just size based on the windwow length of
    # `t2`. `t1` simply needs to have enough datapoints.
    function trajectory_component_errors(t1, t2; window)
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

    estimate = sample.estimate

    x_observation_error, position_observation_error =
        trajectory_component_errors(sample.ground_truth, sample.observation; window = 1:T_obs)
    x_estimation_error, position_estimation_error =
        trajectory_component_errors(sample.ground_truth, estimate; window)

    parameter_estimation_error =
        map(player_cost_models_gt, estimate.player_weights) do cost_model_gt, weights_est
            @assert sum(weights_est) ≈ 1
            parameter_distance(CostUtils.normalize(cost_model_gt.weights), weights_est)
        end |> Statistics.mean

    (;
        sample.idx,
        sample.estimator_name,
        sample.converged,
        sample.σ,
        sample.observation_horizon,
        x_observation_error,
        position_observation_error,
        x_estimation_error,
        position_estimation_error,
        parameter_estimation_error,
    )
end
