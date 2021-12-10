# TODO: maybe allow for different noise levels per dimension (i.e. allow to pass covariance matrix
# here.). But in that case, we would also need to weigh the different dimensions with the correct
# information matrix in the KKT constraint approach.

function generate_dataset_noise_sweep(;
    noise_levels,
    solve_args,
    solve_kwargs = (; solver_attributes = (; print_level = 1)),
    n_observation_sequences_per_instance,
    observation_horizon = solve_args.T,
    rng = Random.MersenneTwister(1),
)
    converged_gt, forward_solution_gt, forward_opt_model_gt =
        solve_game(solve_args...; solve_kwargs...)
    converged_gt || error("Forward solution for the ground truth demonstration did not converge.")

    # add additional time step to make last x relevant state obseravable in the partially observed
    # case
    x_extra = DynamicsModelInterface.next_x(
        solve_args.control_system,
        forward_solution_gt.x[:, end],
        forward_solution_gt.u[:, end],
    )

    # Note: reducing over inner loop to flatten the dataset
    dataset = mapreduce(vcat, noise_levels) do σ
        n_samples = iszero(σ) ? 1 : n_observation_sequences_per_instance
        map(1:n_samples) do _
            (;
                σ,
                observation_horizon,
                ground_truth = forward_solution_gt,
                observation = (;
                    x = forward_solution_gt.x + σ * randn(rng, size(forward_solution_gt.x)),
                    u = forward_solution_gt.u + σ * randn(rng, size(forward_solution_gt.u)),
                    x_extra = x_extra + σ * randn(rng, size(x_extra)),
                ),
            )
        end
    end

    dataset
end

function generate_dataset_observation_window_sweep(; observation_horizons, noise_level, kwargs...)
    ow_end = maximum(observation_horizons)

    mapreduce(vcat, observation_horizons) do observation_horizon
        dataset = generate_dataset_noise_sweep(; kwargs..., noise_levels = [noise_level])

        # computing observation window in a way that aligns them on the right. That is, we want to
        # make sure that for all windows we have to *predict* over the same horizion
        ow_start = ow_end - observation_horizon + 1
        ow = ow_start:ow_end

        map(dataset) do d
            # truncate the forward_solution_gt to start at the first observation index
            gt_truncated =
                (; x = d.ground_truth.x[:, ow_start:end], u = d.ground_truth.u[:, ow_start:end])
            # NOTE: This does not handle the `extra_observation` but it also is not needed in the
            # online experiments due to the different cost structure
            obs_windowed = (; x = d.observation.x[:, ow], u = d.observation.u[:, ow])
            (; d.σ, observation_horizon, ground_truth = gt_truncated, observation = obs_windowed)
        end
    end
end
