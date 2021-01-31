struct Unicycle end

function Base.getproperty(::Unicycle, sym::Symbol)
    if sym === :n_states
        4
    elseif sym === :n_controls
        2
    else
        getfield(x, sym)
    end
end

function DynamicsModelInterface.visualize_trajectory(
    ::Unicycle,
    x;
    canvas = Plots.plot(),
    kwargs...,
)
    (x_min, x_max) = extrema(x[1, :]) .+ (-0.5, 0.5)
    (y_min, y_max) = extrema(x[2, :]) .+ (-0.5, 0.5)

    Plots.plot!(
        canvas,
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
function DynamicsModelInterface.add_dynamics_constraints!(::Unicycle, opt_model, x, u)
    T = size(x)[2]

    # auxiliary variables for nonlinearities
    cosθ = @variable(opt_model, [1:T])
    @NLconstraint(opt_model, [t = 1:T], cosθ[t] == cos(x[4, t]))
    sinθ = @variable(opt_model, [1:T])
    @NLconstraint(opt_model, [t = 1:T], sinθ[t] == sin(x[4, t]))

    @constraint(
        opt_model,
        [t = 1:(T - 1)],
        x[:, t + 1] .== [
            x[1, t] + x[3, t] * cosθ[t],
            x[2, t] + x[3, t] * sinθ[t],
            x[3, t] + u[1, t],
            x[4, t] + u[2, t],
        ]
    )
end

function DynamicsModelInterface.add_dynamics_jacobians!(::Unicycle, opt_model, x, u)
    n_states, T = size(x)
    n_controls = size(u, 1)

    # auxiliary variables for nonlinearities
    cosθ = @variable(opt_model, [1:T])
    @NLconstraint(opt_model, [t = 1:T], cosθ[t] == cos(x[4, t]))
    sinθ = @variable(opt_model, [1:T])
    @NLconstraint(opt_model, [t = 1:T], sinθ[t] == sin(x[4, t]))

    # jacobians of the dynamics in x
    dfdx = @variable(opt_model, [1:n_states, 1:n_states, 1:T])
    @constraint(
        opt_model,
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
