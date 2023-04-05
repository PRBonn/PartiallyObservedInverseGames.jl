# TODO: code duplication could be reduced if both solvers supported the same interface
function estimate(
    solver::InverseKKTConstraintSolver;
    dataset,
    control_system,
    player_cost_models,
    solver_attributes,
    expected_observation = identity,
    estimator_name = (expected_observation === identity ? "Ours Full" : "Ours Partial"),
    solver_kwargs...,
)
    @showprogress pmap(dataset) do d
        observation_model = (; d.σ, expected_observation)

        converged, estimate, opt_model = solve_inverse_game(
            solver,
            expected_observation(d.observation.x);
            control_system,
            observation_model,
            player_cost_models,
            solver_attributes,
            solver_kwargs...,
            # NOTE: This estimator does not use any information beyond the state observation!
        )
        converged || @warn "conKKT did not converge on observation $(d.idx)."

        (; d..., estimate, converged, estimator_name)
    end
end

function estimate(
    solver::AugmentedInverseKKTResidualSolver;
    dataset,
    control_system,
    player_cost_models,
    solver_attributes,
    expected_observation = identity,
    estimator_name = (expected_observation === identity ? "Baseline Full" : "Baseline Partial"),
    solver_kwargs...,
)
    @showprogress pmap(dataset) do d
        observation_model = (; d.σ, expected_observation)

        converged, estimate, opt_model = solve_inverse_game(
            solver,
            expected_observation(d.observation.x);
            control_system,
            observation_model,
            player_cost_models,
            solver_attributes,
            solver_kwargs...,
            # NOTE: This estimator does not use any information beyond the state observation!
        )
        converged || @warn "resKKT did not converge on observation $(d.idx)."

        (; d..., estimate, converged, estimator_name)
    end
end

# TODO: maybe get rid of this (use `AugmentedInverseKKTResidualSolver` instead)
function estimate(
    solver::InverseKKTResidualSolver;
    dataset,
    control_system,
    player_cost_models,
    solver_attributes,
    expected_observation = identity,
    estimator_name = (expected_observation === identity ? "Baseline Full" : "Baseline Partial"),
    pre_solve_kwargs = (;),
)
    @showprogress pmap(dataset) do d
        observation_model = (; d.σ, expected_observation)
        local smooth_model
        smoothed_observation = let
            pre_solve_converged, pre_solve_solution, smooth_model = InversePreSolve.pre_solve(
                # pre-filter for baseline receives one extra state observation to avoid
                # unobservability of the velocity at the end-point.
                # TODO: allow to disable this
                expected_observation([d.observation.x d.observation.x_extra]),
                nothing;
                control_system,
                observation_model,
                solver_attributes,
                pre_solve_kwargs...,
            )
            @assert pre_solve_converged
            # Filtered sequence is truncated to the original length to give all methods the same
            # number of data-points for inference.
            (;
                x = pre_solve_solution.x[:, 1:size(d.observation.x, 2)],
                u = pre_solve_solution.u[:, 1:size(d.observation.u, 2)],
            )
        end

        converged, estimate, opt_model = solve_inverse_game(
            solver,
            smoothed_observation.x,
            smoothed_observation.u;
            control_system,
            player_cost_models,
            solver_attributes,
        )
        converged || @warn "resKKT did not converge on observation $(d.idx)."

        (; d..., estimate, converged, estimator_name, smoothed_observation)
    end
end
