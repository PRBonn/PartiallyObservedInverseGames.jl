using JuMP: JuMP, @NLconstraint, @constraint, @objective, @variable
using LinearAlgebra: I
import Ipopt

include("utils.jl")
#=================================== Test with nonlinear dynamics ==================================#

"""
State layout: x = [px, py, v, θ]'.
"""
function unicycle_dynamics(px, py, v, θ, Δv, Δθ)
    [px + v * cos(θ), py + v * sin(θ), v + Δv, θ + Δθ]
end

# TODO: also allow for nonlinear costs
"The performance index for the forward optimal control problem."
function forward_objective(x, u; Q, R, T)
    sum(x[:, t]' * Q * x[:, t] + u[:, t]' * R * u[:, t] for t in 1:T)
end

# TODO: think about handling of initial guees
# TODO: the number of states and controls could be recovered from the `dynamics` object.
"Solves a forward LQR problem using JuMP."
function solve_optimal_control(dynamics, Q, R, x0, T)
    model = JuMP.Model(Ipopt.Optimizer)

    @variable(model, x[1:(dynamics.n_states), 1:T])
    @variable(model, u[1:(dynamics.n_controls), 1:T])

    # TODO: generalize this to take arbitrary nonlinear, vector-valued functions
    @NLconstraint(
        model,
        dynamics_x1[t = 1:(T - 1)],
        x[1, t + 1] == x[1, t] + x[3, t] * cos(x[4, t])
    )
    @NLconstraint(
        model,
        dynamics_x2[t = 1:(T - 1)],
        x[2, t + 1] == x[2, t] + x[3, t] * sin(x[4, t])
    )
    @NLconstraint(model, dynamics_x3[t = 1:(T - 1)], x[3, t + 1] == x[3, t] + u[1, t])
    @NLconstraint(model, dynamics_x4[t = 1:(T - 1)], x[4, t + 1] == x[4, t] + u[2, t])
    @constraint(model, initial_condition, x[:, 1] .== x0)
    @objective(model, Min, forward_objective(x, u; Q, R, T))
    JuMP.optimize!(model)
    get_model_values(model, :x, :u), model
end

unicycle_solution, forward_model = let
    dynamics = (next_x = unicycle_dynamics, n_states = 4, n_controls = 2)
    Q = I
    R = 100I
    x0 = [1, 1, 0, 0]
    T = 100
    solve_optimal_control(dynamics, Q, R, x0, T)
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
