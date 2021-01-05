# TODO: Implement a simple converter for ControlSystems.jl

#============================================ Preamble =============================================#

import Zygote
import Ipopt
using JuMP: JuMP, @constraint, @objective, @variable
using LinearAlgebra: I
using Test: @test, @testset

include("utils.jl")

T = 100
A = repeat([[
    1 1
    0 1
]], T)
B = repeat([[0, 1][:, :]], T)
Q = I
R = 100I
x0 = [10.0, 10.0]

#============================================== Utils ==============================================#

function dynamics_constraints(x, u; A, B)
    reduce(hcat, ((x[:, t + 1] - A[t] * x[:, t] - B[t] * u[:, t]) for t in axes(x)[2][1:(end - 1)]))
end

#=========================================== Forward LQR ===========================================#

"Solves a forward LQR problem using JuMP."
function solve_lqr(A, B, Q, R, x0, T)
    n_states, n_controls = size(only(unique(B)))
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, x[1:n_states, 1:T])
    @variable(model, u[1:n_controls, 1:T])
    @constraint(model, dynamics, dynamics_constraints(x, u; A, B) .== 0)
    @constraint(model, initial_condition, x[:, 1] .== x0)
    @objective(model, Min, forward_objective(x, u; Q, R))
    JuMP.optimize!(model)
    get_model_values(model, :x, :u), model
end

lqr_solution, _ = solve_lqr(A, B, Q, R, x0, T)

#=========================================== Inverse LQR ===========================================#

"The lagrangian of the forward LQR problem."
function lqr_lagrangian(x, u, λ; Q, R, A, B)
    c_dyn = dynamics_constraints(x, u; A, B)
    @views forward_objective(x, u; Q, R) +
           sum(λ[:, t]' * c_dyn[:, t] for t in axes(x)[2][1:(end - 1)])
end

"The hand-written gradient of the forward LQR problem in x."
function lqr_lagrangian_grad_x(x, u, λ; Q, R, A, B)
    hcat(
        2 * Q * x[:, 1] - A[1]' * λ[:, 1], # special handling of first time step
        reduce(
            hcat,
            (
                2 * Q * x[:, t + 1] + λ[:, t] - A[t + 1]' * λ[:, t + 1]
                for t in axes(x)[2][1:(end - 1)]
            ),
        ),
    )
end

"The hand-written gradient of the forward LQR problem in u."
function lqr_lagrangian_grad_u(x, u, λ; Q, R, A, B)
    reduce(hcat, (2 * R * u[:, t] - B[t]' *  λ[:, t] for t in axes(x)[2]))
end

"""
Solves aninverse LQR problem using JuMP.

`x̂` is the (observed) state trajectory for which we wish to find the cost parametrization.

`Q̃` and `R̃` are iterables of quadatic state and control cost matrices for which the weight
vectors`q` and `r` are to be estimated.
"""
function solve_inverse_lqr(x̂, Q̃, R̃; A, B, r_sqr_min = 1e-5)
    T = size(x̂)[2]
    n_states, n_controls = size(only(unique(B)))
    model = JuMP.Model(Ipopt.Optimizer)

    # decision variable
    @variable(model, q[1:length(Q̃)] >= 0)
    @variable(model, r[1:length(R̃)] >= 0)
    @variable(model, x[1:n_states, 1:T])
    @variable(model, u[1:n_controls, 1:T])
    @variable(model, λ[1:n_states, 1:T]) # multipliers of the forward LQR condition

    # initial condition
    @constraint(model, initial_condition, x[:, 1] .== x̂[:, 1])
    # dynamics
    @constraint(model, dynamics_constraints(x, u; A, B) .== 0)
    # Optimality conditions (KKT) of forward LQR show up as a constraints
    Q = sum(q .* Q̃)
    R = sum(r .* R̃)
    @constraint(model, ∇ₓL, lqr_lagrangian_grad_x(x, u, λ; Q, R, A, B)[:, 2:end] .== 0)
    @constraint(model, ∇ᵤL, lqr_lagrangian_grad_u(x, u, λ; Q, R, A, B) .== 0)
    # regularization
    @constraint(model, r' * r >= r_sqr_min)
    @constraint(model, r' * r + q' * q == 1)

    @objective(model, Min, inverse_objective(x; x̂))
    JuMP.optimize!(model)
    get_model_values(model, :q, :r, :x, :u, :λ), model, JuMP.value.(Q), JuMP.value.(R)
end

#====================== Inverse LQR as nested constrained optimization problem =====================#

@testset "Inverse LQR" begin
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
    ilqr_solution, ilqr_model, Q_est, R_est = solve_inverse_lqr(lqr_solution.x, Q̃, R̃; A, B)
    ∇ₓL_sol = JuMP.value.(ilqr_model[:∇ₓL])
    ∇ᵤL_sol = JuMP.value.(ilqr_model[:∇ᵤL])

    @testset "Gradient Check" begin
        grad_args = (ilqr_solution.x, ilqr_solution.u, ilqr_solution.λ)
        grad_kwargs = (; Q = Q_est, R = R_est, A, B)

        ∇ₓL_ad, ∇ᵤL_ad = Zygote.gradient(
            (x, u) -> lqr_lagrangian(x, u, ilqr_solution.λ; grad_kwargs...),
            ilqr_solution.x,
            ilqr_solution.u,
        )
        ∇ₓL_manual = lqr_lagrangian_grad_x(grad_args...; grad_kwargs...)
        ∇ᵤL_manual = lqr_lagrangian_grad_u(grad_args...; grad_kwargs...)
        atol = 1e-10

        @test isapprox(∇ₓL_ad, ∇ₓL_manual; atol = atol)
        @test isapprox(∇ₓL_ad[:, 2:end], ∇ₓL_sol; atol = atol)
        @test isapprox(∇ᵤL_ad, ∇ᵤL_manual; atol = atol)
        @test isapprox(∇ᵤL_ad, ∇ᵤL_sol; atol = atol)
    end

    @testset "Solution Sanity" begin
        @test JuMP.termination_status(ilqr_model) in (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL)
        @test Q_est[1, 1] / Q_est[2, 2] ≈ Q[1, 1] / Q[2, 2]
        @test Q_est[1, 1] / R_est[1, 1] ≈ Q[1, 1] / R[1, 1]
        @test Q_est[2, 2] / R_est[1, 1] ≈ Q[2, 2] / R[1, 1]
        @test ilqr_solution.x ≈ first(solve_lqr(A, B, Q_est, R_est, x0, T)).x
    end
end
