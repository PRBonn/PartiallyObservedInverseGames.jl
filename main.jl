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

#============= Describe a simple forward LQR problem using a collocation style method  =============#
lqr_problem = let
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, x[1:n_states, 1:T])
    @variable(model, u[1:n_controls, 1:T])
    @constraint(model, initial_condition, x[:, 1] .== x0)
    @constraint(model, dynamics[t = 1:(T - 1)], x[:, t + 1] .== A * x[:, t] + B * u[:, t])
    JuMP.@objective(model, Min, sum(x[:, t]' * Q * x[:, t] + u[:, t]' * R * u[:, t] for t in 1:T))
    JuMP.optimize!(model)
    model
end

x̂ = JuMP.value.(JuMP.value.(lqr_problem[:x]))

#====================== Inverse LQR as nested constrained optimization problem =====================#

Q̃ = 0.001I
R̃ = R

inverse_lqr_problem = let
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, q̃)
    @variable(model, r̃ == 1)               # TODO: fixed for now
    @variable(model, x̃0[1:n_states])       # initial condition
    @variable(model, x̃[1:n_states, 1:T])   # state trajectory
    @variable(model, ũ[1:n_controls, 1:T]) # input trajectory
    @constraint(model, initial_condition, x̃[:, 1] .== x̃0)
    @constraint(model, dynamics[t = 1:(T - 1)], x̃[:, t + 1] .== A * x̃[:, t] + B * ũ[:, t])

    # Optimality conditions (KKT) of forward LQR show up as a constraints
    @variable(model, λ̃[1:n_states, 1:T]) # multipliers of the forward LQR condition
    @constraint(
        model,
        lqr_lagrangian_grad_x[t = 1:(T - 1)],
        2q̃ * Q̃ * x̃[:, t + 1] + λ̃[:, t] - (λ̃[:, t + 1]' * A)' .== 0
    )
    @constraint(model, lqr_lagrangian_grad_u[t = 1:T], 2r̃ * R̃ * ũ[:, t] - (λ̃[:, t]' * B)' .== 0)

    # TODO: figure out why regularization breaks things here??? Did I forget any constraints?
    @objective(model, Min, sum(sum((x̂[:, t] - x̃[:, t]) .^ 2) for t in 1:T)) # + (q̃^2 + r̃^2))
    model
end

JuMP.optimize!(inverse_lqr_problem)
@test JuMP.value(inverse_lqr_problem[:q̃]) * Q̃ ≈ Q
