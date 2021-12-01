module InversePreSolve

import PartiallyObservedInverseGames.ForwardOptimalControl
using JuMP: @objective

export pre_solve

function pre_solve(
    y_obs,
    u_obs;
    control_system,
    observation_model = (; expected_observation = identity),
    T = size(y_obs, 2),
    u_regularization = 0,
    inner_solver_kwargs...,
)
    T >= size(y_obs, 2) ||
        throw(ArgumentError("Horizon `T` must be at least as long as number of observations."))
    function presolve_objective(; x, u, y_obs, u_obs)
        y_expected = observation_model.expected_observation(x)
        sum(el -> el^2, y_expected[:, 1:size(y_obs, 2)] - y_obs) +
        (isnothing(u_obs) ? 0 : sum(el -> el^2, u - u_obs)) +
        (iszero(u_regularization) ? 0 : u_regularization * sum(el -> el^2, u))
    end

    reconstruction_cost_model = (;
        # TODO: make configurable perhaps
        add_objective! = function (opt_model, x, u; kwargs...)
            @objective(opt_model, Min, presolve_objective(; x, u, y_obs, u_obs))
        end,
        weights = (),
    )

    ForwardOptimalControl.solve_optimal_control(
        control_system,
        reconstruction_cost_model,
        nothing,
        T;
        inner_solver_kwargs...,
    )
end

end
