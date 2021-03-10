import Zygote
import JuMP

using Test: @test, @testset
using PartiallyObservedInverseGames.ForwardOptimalControl: solve_lqr
using PartiallyObservedInverseGames.InverseOptimalControl:
    solve_inverse_lqr, lqr_lagrangian, lqr_lagrangian_grad_x, lqr_lagrangian_grad_u
using LinearAlgebra: I

#=========================================== Forward LQR ===========================================#

T = 100
A = repeat([[
    1 1
    0 1
]], T)
B = repeat([[0, 1][:, :]], T)
Q = I
R = 100I
x0 = [10.0, 10.0]

forward_converged, forward_solution, forward_model = solve_lqr(A, B, Q, R, x0, T)

@testset "Forward LQR" begin
    @test forward_converged
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
    inverse_converged, inverse_solution, inverse_model, Q_est, R_est =
        solve_inverse_lqr(forward_solution.x, Q̃, R̃; A, B)

    @testset "Gradient Check" begin
        grad_args = (inverse_solution.x, inverse_solution.u, inverse_solution.λ)
        grad_kwargs = (; Q = Q_est, R = R_est, A, B)

        ∇ₓL_ad, ∇ᵤL_ad = Zygote.gradient(
            (x, u) -> lqr_lagrangian(x, u, inverse_solution.λ; grad_kwargs...),
            inverse_solution.x,
            inverse_solution.u,
        )
        ∇ₓL_manual = lqr_lagrangian_grad_x(grad_args...; grad_kwargs...)
        ∇ᵤL_manual = lqr_lagrangian_grad_u(grad_args...; grad_kwargs...)
        atol = 1e-10

        @test isapprox(∇ₓL_ad, ∇ₓL_manual; atol = atol)
        @test isapprox(∇ₓL_ad[:, 2:end], inverse_solution.∇ₓL; atol = atol)
        @test isapprox(∇ᵤL_ad, ∇ᵤL_manual; atol = atol)
        @test isapprox(∇ᵤL_ad, inverse_solution.∇ᵤL; atol = atol)
    end

    @testset "Solution Sanity" begin
        @test inverse_converged
        @test Q_est[1, 1] / Q_est[2, 2] ≈ Q[1, 1] / Q[2, 2]
        @test Q_est[1, 1] / R_est[1, 1] ≈ Q[1, 1] / R[1, 1]
        @test Q_est[2, 2] / R_est[1, 1] ≈ Q[2, 2] / R[1, 1]
        @test inverse_solution.x ≈ solve_lqr(A, B, Q_est, R_est, x0, T)[2].x
    end
end
