function estimator_statistics(
    sample;
    player_cost_models_gt,
    position_indices,
    state_distance = Distances.euclidean,
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

    function trajectory_component_errors(t1, t2; window)
        if haskey(t2, :x)
            n_players = length(player_cost_models_gt)
            @assert n_players == 2
            grouped_position_indices = Iterators.partition(position_indices, n_players) |> collect
            # mean over players
            Statistics.mean(grouped_position_indices) do pindex
                t1_states = t1.x[pindex, window]
                t2_states = t2.x[pindex, window]
                # mean over time
                Statistics.mean(Distances.colwise(state_distance, t1_states, t2_states))
            end
        else
            Inf
        end
    end

    estimate = sample.estimate

    position_observation_error =
        trajectory_component_errors(sample.ground_truth, sample.observation; window = 1:T_obs)
    position_estimation_error = trajectory_component_errors(sample.ground_truth, estimate; window)

    parameter_estimation_error =
        map(player_cost_models_gt, estimate.player_weights) do cost_model_gt, weights_est
            @assert sum(weights_est) ≈ 1
            parameter_distance(CostUtils.normalize(cost_model_gt.weights), weights_est)
        end |> Statistics.mean

    observation_model_type = contains(sample.estimator_name, "Partial") ? "Partial" : "Full"

    (;
        sample.idx,
        sample.estimator_name,
        sample.converged,
        sample.σ,
        sample.observation_horizon,
        position_observation_error,
        position_estimation_error,
        parameter_estimation_error,
        observation_model_type,
    )
end
