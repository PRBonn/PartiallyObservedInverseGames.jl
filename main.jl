# TODO: Make time-varying
# TODO: Implement a simple converter for ControlSystems.jl

#============================================ Preamble =============================================#

import JuMP: JuMP, @constraint, @objective, @variable
import Ipopt
using LinearAlgebra: I
using Test: @test, @testset

T = 100
n_states = 2
n_controls = 1
A = [
    1 1
    0 1
]
B = [0, 1][:, :]
Q = I
R = 100I
x0 = [10.0, 10.0]

#============================================== Utils ==============================================#

function add_dynamics!(model, A, B)
    @variable(model, x[1:n_states, 1:T])
    @variable(model, u[1:n_controls, 1:T])
    @constraint(model, dynamics[t = 1:(T - 1)], x[:, t + 1] .== A * x[:, t] + B * u[:, t])
    x, u, dynamics
end

get_model_values(model, symbols...) = (; map(sym -> sym => JuMP.value.(model[sym]), symbols)...)

#=========================================== Forward LQR ===========================================#

"The performance index for the forward optimal control problem."
function forward_objective(x, u; Q, R)
    sum(x[:, t]' * Q * x[:, t] + u[:, t]' * R * u[:, t] for t in 1:T)
end

"Solves a forward LQR problem using JuMP."
function solve_lqr(A, B, Q, R, x0)
    model = JuMP.Model(Ipopt.Optimizer)
    x, u, _ = add_dynamics!(model, A, B)
    @constraint(model, initial_condition, x[:, 1] .== x0)
    @objective(model, Min, forward_objective(x, u; Q, R))
    JuMP.optimize!(model)
    get_model_values(model, :x, :u), model
end

lqr_solution, _ = solve_lqr(A, B, Q, R, x0)

#=========================================== Inverse LQR ===========================================#

"The performance index for the inverse optimal control problem."
function inverse_objective(x, q, r; x̂)
    sum(sum((x̂[:, t] - x[:, t]) .^ 2) for t in 1:T)
end

"The lagrangian of the forward LQR problem."
function lqr_lagrangian(x, u, λ; Q, R, A, B)
    forward_objective(x, u; Q, R) +
    sum(λ[:, t]' * (x[:, t + 1] - A * x[:, t] - B * u[:, t]) for t in 1:(T - 1))
end

"The hand-written gradient of the forward LQR problem in x."
function lqr_lagrangian_grad_x(x, u, λ; Q, R, A, B, T)
    hcat(
        2 * Q * x[:, 1] - A' * λ[:, 1], # special handling of first time step
        reduce(hcat, (2 * Q * x[:, t + 1] + λ[:, t] - (λ[:, t + 1]' * A)' for t in 1:(T - 1))),
    )
end

"The hand-written gradient of the forward LQR problem in u."
function lqr_lagrangian_grad_u(x, u, λ; Q, R, A, B, T)
    reduce(hcat, (2 * R * u[:, t] - (λ[:, t]' * B)' for t in 1:T))
end

"""
Solves aninverse LQR problem using JuMP.

`x̂` is the (observed) state trajectory for which we wish to find the cost parametrization.

`Q̃` and `R̃` are iterables of quadatic state and control cost matrices for which the weight
vectors`q` and `r` are to be estimated.
"""
function solve_inverse_lqr(x̂, A, B, Q̃, R̃)
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, q[1:length(Q̃)] >= 0)
    @variable(model, r[1:length(R̃)] >= 0)
    @constraint(model, r' * r >= 1e-5)
    @constraint(model, r' * r + q' * q == 1)
    x, u, _ = add_dynamics!(model, A, B)
    @constraint(model, initial_condition, x[:, 1] .== x̂[:, 1])

    Q = sum(q .* Q̃)
    R = sum(r .* R̃)

    # Optimality conditions (KKT) of forward LQR show up as a constraints
    @variable(model, λ[1:n_states, 1:T]) # multipliers of the forward LQR condition
    @constraint(
        model,
        lqr_lagrangian_dx,
        lqr_lagrangian_grad_x(x, u, λ; Q, R, A, B, T)[:, 2:end] .== 0
    )
    @constraint(model, lqr_lagrangian_du, lqr_lagrangian_grad_u(x, u, λ; Q, R, A, B, T) .== 0)

    # TODO: figure out why regularization breaks things here??? Did I forget any constraints?
    @objective(model, Min, inverse_objective(x, q, r; x̂))
    JuMP.optimize!(model)
    get_model_values(model, :q, :r, :x, :u, :λ), model, JuMP.value.(Q), JuMP.value.(R)
end
#====================== Inverse LQR as nested constrained optimization problem =====================#

Q̃ = [
    [
        1//3 0
        0 0
    ],
    [
        0 0
        0 2//3
    ],
]
R̃ = [R]
ilqr_solution, ilqr_model, Q_est, R_est = solve_inverse_lqr(lqr_solution.x, A, B, Q̃, R̃)

@testset "Inverse LQR" begin
    @test Q_est[1, 1] / Q_est[2, 2] ≈ Q[1, 1] / Q[2, 2]
    @test Q_est[1, 1] / R_est[1, 1] ≈ Q[1, 1] / R[1, 1]
    @test Q_est[2, 2] / R_est[1, 1] ≈ Q[2, 2] / R[1, 1]
    @test ilqr_solution.x ≈ first(solve_lqr(A, B, Q_est, R_est, x0)).x
end
