module InversePreSolve

import JuMPOptimalControl.ForwardOptimalControl
using JuMP: @objective

export pre_solve

function pre_solve(
    y_obs,
    u_obs;
    control_system,
    observation_model = (; expected_observation = identity),
    inner_solver_kwargs...,
)
    function presolve_objective(; x, u, y_obs, u_obs)
        y_expected = observation_model.expected_observation(x)
        sum(el -> el^2, y_expected - y_obs) + (isnothing(u_obs) ? 0 : sum(el -> el^2, u - u_obs))
    end

    reconstruction_cost_model = (;
        # TODO: make configurable perhaps
        add_objective! = function (opt_model, x, u; kwargs...)
            @objective(opt_model, Min, presolve_objective(; x, u, y_obs, u_obs))
        end,
        weights = (),
    )
    T = size(y_obs, 2)

    ForwardOptimalControl.solve_optimal_control(
        control_system,
        reconstruction_cost_model,
        nothing,
        T;
        inner_solver_kwargs...,
    )
end

end
