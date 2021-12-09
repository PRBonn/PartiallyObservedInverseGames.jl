# TODO: maybe allow for different noise levels per dimension (i.e. allow to pass covariance matrix
# here.). But in that case, we would also need to weigh the different dimensions with the correct
# information matrix in the KKT constraint approach.

function generate_dataset(;
    solve_args,
    solve_kwargs = (; solver_attributes = (; print_level = 1)),
    noise_levels,
    n_observation_sequences_per_noise_level,
    observation_window = 1:(solve_args.T),
    rng = Random.MersenneTwister(1),
)
    converged_gt, forward_solution_gt, forward_opt_model_gt =
        solve_game(solve_args...; solve_kwargs...)
    @assert converged_gt

    forward_solution_gt_truncated = (;
        x = forward_solution_gt.x[:, (observation_window[begin]):end],
        u = forward_solution_gt.u[:, (observation_window[begin]):end],
    )

    forward_solution_gt_observation_window = (;
        x = forward_solution_gt.x[:, observation_window],
        u = forward_solution_gt.u[:, observation_window],
    )

    # add additional time step to make last x relevant state obseravable in the partially observed
    # case
    x_extra = DynamicsModelInterface.next_x(
        solve_args.control_system,
        forward_solution_gt_observation_window.x[:, end],
        forward_solution_gt_observation_window.u[:, end],
    )

    # Note: reducing over inner loop to flatten the dataset
    dataset = mapreduce(vcat, noise_levels) do σ
        n_samples = iszero(σ) ? 1 : n_observation_sequences_per_noise_level
        observations = map(1:n_samples) do _
            (;
                σ,
                x = forward_solution_gt_observation_window.x +
                    σ * randn(rng, size(forward_solution_gt_observation_window.x)),
                u = forward_solution_gt_observation_window.u +
                    σ * randn(rng, size(forward_solution_gt_observation_window.u)),
                x_extra = x_extra + σ * randn(rng, size(x_extra)),
            )
        end
    end

    forward_solution_gt_truncated, dataset
end
