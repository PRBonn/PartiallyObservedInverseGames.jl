
d = dataset[end]
observation_horizon = T ÷ 3
y = d.x[:, (1:observation_horizon) .+ 5]
observation_model = (; d.σ, expected_observation = identity)

solver = InverseKKTResidualSolver()

# TODO: do this in a cleaner way. Would be nice to have the residual solver implement the exact same
# interface as the constraint solver by just doing pre-filtering etc inside the one-argument version
# of `solve_inverse_game`. Could probably just be a dispatch.
converged, sol = if solver isa InverseKKTConstraintSolver
    solve_inverse_game(
        solver,
        y;
        control_system,
        observation_model,
        player_cost_models = player_cost_models_gt,
        T,
        cmin = 1e-3,
        player_weight_prior = nothing, #[ones(4) / 4 for _ in 1:2],
        pre_solve_kwargs = (; u_regularization = 1e-5),
        max_observation_error = nothing,
    )
elseif solver isa InverseKKTResidualSolver
    smoothed_observation = let
        pre_solve_converged, pre_solve_solution = InversePreSolve.pre_solve(
            # pre-filter for baseline receives one extra state observation to avoid
            # unobservability of the velocity at the end-point.
            y,
            nothing;
            control_system,
            observation_model,
        )
        @assert pre_solve_converged
        # Filtered sequence is truncated to the original length to give all methods the same
        # number of data-points for inference.
        (; x = pre_solve_solution.x[:, 1:size(y, 2)], u = pre_solve_solution.u[:, 1:size(y, 2)])
    end
    converged, estimate = solve_inverse_game(
        solver,
        smoothed_observation.x,
        smoothed_observation.u;
        control_system,
        player_cost_models = player_cost_models_gt,
    )

    sol = merge(estimate, (; converged, smoothed_observation...))
    converged, sol
else
    error("Unknown solver.")
end
@assert converged

# visualization
let
    gt = TrajectoryVisualization.trajectory_data(control_system, dataset[begin].x)
    observation = TrajectoryVisualization.trajectory_data(control_system, y)
    estimate = TrajectoryVisualization.trajectory_data(control_system, sol.x)

    canvas = TrajectoryVisualization.visualize_trajectory(gt; group = "ground truth", legend = true)
    canvas =
        TrajectoryVisualization.visualize_trajectory(observation; canvas, group = "observation")
    canvas = TrajectoryVisualization.visualize_trajectory(estimate; canvas, group = "estimation")
    display(canvas)
end

function parameter_error(ps1, ps2)
    map(ps1, ps2) do p1, p2
        p1 = CostUtils.normalize(p1)
        p2 = CostUtils.normalize(p2)
        Distances.cosine_dist(p1, p2)
    end |> Statistics.mean
end

ws_gt = [m.weights for m in player_cost_models_gt]
@show parameter_error(ws_gt, sol.player_weights)
