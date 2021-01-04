using JuMP: JuMP, @NLconstraint, @constraint, @objective, @variable
using LinearAlgebra: I
import Ipopt

include("utils.jl")
#=================================== Test with nonlinear dynamics ==================================#

"The performance index for the forward optimal control problem."
function forward_objective(x, u; Q, R, T)
    sum(x[:, t]' * Q * x[:, t] + u[:, t]' * R * u[:, t] for t in 1:T)
end


# TODO: think about handling of initial guees
"Solves a forward LQR problem using JuMP."
function solve_optimal_control(control_system, Q, R, x0, T)
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, x[1:(control_system.n_states), 1:T])
    @variable(model, u[1:(control_system.n_controls), 1:T])

    # TODO: generalize this to take arbitrary nonlinear, vector-valued functions NOTE: This seems to
    # be non-trivial to generalize; This would require some meta programming macromagic or generated
    # function perhaps.
    #
    # These constraints encode the dynamics of a unicycle with state layout x_t = [px, py, v, θ] and
    # inputs u_t = [Δv, Δθ].
    begin
        # px
        @NLconstraint(
            model,
            dynamics_px[t = 1:(T - 1)],
            x[1, t + 1] == x[1, t] + x[3, t] * cos(x[4, t])
        )
        # py
        @NLconstraint(
            model,
            dynamics_py[t = 1:(T - 1)],
            x[2, t + 1] == x[2, t] + x[3, t] * sin(x[4, t])
        )
        # Δv, Δθ
        @constraint(model, dynamics_vθ[1:2, t = 1:(T - 1)], x[3:4, t + 1] .== x[3:4, t] .+ u[1:2, t])
    end

    @constraint(model, initial_condition, x[:, 1] .== x0)
    @objective(model, Min, forward_objective(x, u; Q, R, T))
    JuMP.optimize!(model)
    get_model_values(model, :x, :u), model
end

unicycle_solution, forward_model = let
    control_system = (n_states = 4, n_controls = 2)
    Q = I
    R = 100I
    x0 = [1, 1, 0, 0]
    T = 100
    solve_optimal_control(control_system, Q, R, x0, T)
end

# Visualization

import ElectronDisplay
import Plots
Plots.gr()
Plots.theme(:vibrant)

unicycle_viz = Plots.plot(
    unicycle_solution.x[1, :],
    unicycle_solution.x[2, :],
    quiver = (
        unicycle_solution.x[3, :] .* cos.(unicycle_solution.x[4, :]),
        unicycle_solution.x[3, :] .* sin.(unicycle_solution.x[4, :]),
    ),
    st = :quiver,
)
