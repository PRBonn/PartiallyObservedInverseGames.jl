module Unicycle

import Plots
using JuMP: @variable, @constraint, @NLconstraint

export visualize_unicycle_trajectory,
    add_unicycle_dynamics_jacobians!, add_unicycle_dynamics_jacobians!

function visualize_unicycle_trajectory(x; cost_model = nothing, kwargs...)
    (x_min, x_max) = extrema(x[1, :]) .+ (-0.5, 0.5)
    (y_min, y_max) = extrema(x[2, :]) .+ (-0.5, 0.5)

    # TODO: this is a rather ugly hack for quick visualization of the proximity cost
    prox_cost = if !isnothing(cost_model) && haskey(cost_model.weights, :state_proximity)
        Plots.plot(
            x_min:0.01:x_max,
            y_min:0.01:y_max,
            (x, y) ->
                cost_model.weights.state_proximity *
                -log(sum(((x, y) .- cost_model.obstacle) .^ 2)),
            st = :contour,
        )
    else
        Plots.plot()
    end

    unicycle_viz = Plots.plot!(
        prox_cost,
        x[1, :],
        x[2, :];
        quiver = (abs.(x[3, :]) .* cos.(x[4, :]), abs.(x[3, :]) .* sin.(x[4, :])),
        # line_z = axes(x)[2],
        st = :quiver,
        kwargs...,
    )
end

# These constraints encode the dynamics of a unicycle with state layout x_t = [px, py, v, θ] and
# inputs u_t = [Δv, Δθ].
function add_unicycle_dynamics_constraints!(model, x, u)
    T = size(x)[2]

    # auxiliary variables for nonlinearities
    @variable(model, cosθ[1:T])
    @NLconstraint(model, [t = 1:T], cosθ[t] == cos(x[4, t]))

    @variable(model, sinθ[1:T])
    @NLconstraint(model, [t = 1:T], sinθ[t] == sin(x[4, t]))

    @constraint(
        model,
        dynamics[t = 1:(T - 1)],
        x[:, t + 1] .== [
            x[1, t] + x[3, t] * cosθ[t],
            x[2, t] + x[3, t] * sinθ[t],
            x[3, t] + u[1, t],
            x[4, t] + u[2, t],
        ]
    )
end

function add_unicycle_dynamics_jacobians!(model, x, u)
    n_states, T = size(x)
    n_controls = size(u, 1)
    # TODO it's a bit ugly that we rely on these constraints to be present. We could check with
    # `haskey`.
    cosθ = model[:cosθ]
    sinθ = model[:sinθ]

    # jacobians of the dynamics in x
    @variable(model, dfdx[1:n_states, 1:n_states, 1:T])
    @constraint(
        model,
        [t = 1:T],
        dfdx[:, :, t] .== [
            1 0 cosθ[t] -x[3, t]*sinθ[t]
            0 1 sinθ[t] +x[3, t]*cosθ[t]
            0 0 1 0
            0 0 0 1
        ]
    )

    # jacobians of the dynamics in u
    dfdu = [
        0 0
        0 0
        1 0
        0 1
    ] .* reshape(ones(T), 1, 1, :)

    (; dx = dfdx, du = dfdu)
end

end
