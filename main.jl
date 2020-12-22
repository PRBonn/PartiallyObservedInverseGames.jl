# TODO: Implement a simple converter for ControlSystems.jl
import JuMP: JuMP, @constraint, @objective, @variable
import Ipopt
using LinearAlgebra: I
using Test: @test

T = 100
n_states = 2
n_controls = 1
A = [1 1; 0 1]
B = collect([0 1]')
Q = I
R = 100I
x0 = [10.0, 10.0]

function add_dynamics!(model, A, B, x0)
    @variable(model, x[1:n_states, 1:T])
    @variable(model, u[1:n_controls, 1:T])
    @constraint(model, initial_condition, x[:, 1] .== x0)
    @constraint(model, dynamics[t = 1:(T - 1)], x[:, t + 1] .== A * x[:, t] + B * u[:, t])
    x, u, initial_condition, dynamics
end

#============= Describe a simple forward LQR problem using a collocation style method  =============#
lqr_problem = let
    model = JuMP.Model(Ipopt.Optimizer)
    x, u, _ = add_dynamics!(model, A, B, x0)
    JuMP.@objective(model, Min, sum(x[:, t]' * Q * x[:, t] + u[:, t]' * R * u[:, t] for t in 1:T))
    JuMP.optimize!(model)
    model
end

x̂_lqr = JuMP.value.(JuMP.value.(lqr_problem[:x]))

#====================== Inverse LQR as nested constrained optimization problem =====================#

Q̃ = 100I
R̃ = R

inverse_lqr_problem = let
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, q̃)
    @variable(model, r̃ == 1) # TODO: fixed for now
    x, u, _ = add_dynamics!(model, A, B, x0)

    # Optimality conditions (KKT) of forward LQR show up as a constraints
    @variable(model, λ̃[1:n_states, 1:T]) # multipliers of the forward LQR condition
    @constraint(
        model,
        lqr_lagrangian_dx[t = 1:(T - 1)],
        2q̃ * Q̃ * x[:, t + 1] + λ̃[:, t] - (λ̃[:, t + 1]' * A)' .== 0
    )
    @constraint(model, lqr_lagrangian_du[t = 1:T], 2r̃ * R̃ * u[:, t] - (λ̃[:, t]' * B)' .== 0)

    # TODO: figure out why regularization breaks things here??? Did I forget any constraints?
    @objective(model, Min, sum(sum((x̂_lqr[:, t] - x[:, t]) .^ 2) for t in 1:T)) # + (q̃^2 + r̃^2))
    model
end

JuMP.optimize!(inverse_lqr_problem)
@test JuMP.value(inverse_lqr_problem[:q̃]) * Q̃ ≈ Q

# TODO: plot the result of the inverse controller.
