using Test: @test, @testset

# Optimization
using JuMP: JuMP, @NLconstraint, @constraint, @objective, @variable
using LinearAlgebra: I, diagm
import Ipopt

# Visualization
import ElectronDisplay
import Plots
Plots.gr()
Plots.theme(:vibrant)

include("utils.jl")

#======================================== Global parameters ========================================#

# TODO: generalize this to take arbitrary nonlinear, vector-valued functions NOTE: This seems to be
# non-trivial to generalize; This would require some meta programming macromagic or generated
# function perhaps.
#
# These constraints encode the dynamics of a unicycle with state layout x_t = [px, py, v, θ] and
# inputs u_t = [Δv, Δθ].
function unicycle_dynamics_constraints!(model, x, u)
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

    cosθ, sinθ
end

control_system =
    (dynamics_constraints! = unicycle_dynamics_constraints!, n_states = 4, n_controls = 2)
Q = I
R = 100I
x0 = [1, 1, 0, 0]
T = 100

#====================================== forward optimal control ====================================#

# TODO: think about handling of initial guees
"Solves a forward LQR problem using JuMP."
function solve_optimal_control(
    control_system,
    Q,
    R,
    x0,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (;),
    silent = true,
)
    model = JuMP.Model(solver)
    set_solver_attributes!(model; silent, solver_attributes...)

    @variable(model, x[1:(control_system.n_states), 1:T])
    @variable(model, u[1:(control_system.n_controls), 1:T])
    control_system.dynamics_constraints!(model, x, u)
    @constraint(model, initial_condition, x[:, 1] .== x0)
    @objective(model, Min, forward_objective(x, u; Q, R))
    @time JuMP.optimize!(model)
    get_model_values(model, :x, :u), model
end

function visualize_unicycle_trajectory(x)
    unicycle_viz = Plots.plot(
        x[1, :],
        x[2, :],
        quiver = (abs.(x[3, :]) .* cos.(x[4, :]), abs.(x[3, :]) .* sin.(x[4, :])),
        line_z = axes(x)[2],
        st = :quiver,
    )
end

forward_solution, forward_model = solve_optimal_control(control_system, Q, R, x0, T)
visualize_unicycle_trajectory(forward_solution.x)

#===================================== Inverse Optimal Control =====================================#

function solve_inverse_optimal_control(
    x̂,
    Q̃,
    R̃;
    control_system,
    r_sqr_min = 1e-5,
    solver = Ipopt.Optimizer,
    solver_attributes = (;),
    silent = true,
)
    T = size(x̂)[2]
    model = JuMP.Model(solver)
    set_solver_attributes!(model; silent, solver_attributes...)

    # decision variable
    @variable(model, q[1:length(Q̃)] >= 0)
    @variable(model, r[1:length(R̃)] >= 0)
    @variable(model, x[1:(control_system.n_states), 1:T])
    @variable(model, u[1:(control_system.n_controls), 1:T])
    @variable(model, λ[1:(control_system.n_states), 1:T]) # multipliers of the forward optimality condition

    # initial condition
    @constraint(model, initial_condition, x[:, 1] .== x̂[:, 1])
    # TODO: introduce less dirty hack
    cosθ, sinθ = control_system.dynamics_constraints!(model, x, u)

    # Optimality conditions (KKT) of forward LQR show up as a constraints
    Q = sum(q .* Q̃)
    R = sum(r .* R̃)

    # Auxiliary variable trick (https://jump.dev/JuMP.jl/v0.19/nlp/)
    # TODO: For non-quadratic costs this will have to be different
    begin
        # partials of the cost in x
        @variable(model, dgdx[1:(control_system.n_states), 1:T])
        @constraint(model, dgdx .== 2 * Q * x)
        # partials of the cost in u
        @variable(model, dgdu[1:(control_system.n_controls), 1:T])
        @constraint(model, dgdu .== 2 * R * u)

        # jacobians of the dynamics in x
        @variable(model, dfdx[1:(control_system.n_states), 1:(control_system.n_states), 1:T])
        @constraint(
            model,
            [t = 1:T],
            dfdx[:, :, t] .== [
                1 0 cosθ[t] -x[3, t]*sinθ[t]
                0 1 sinθ[t] x[3, t]*cosθ[t]
                0 0 1 0
                0 0 0 1
            ]
        )
        # jacobians of the dynamics in u
        @variable(model, dfdu[1:(control_system.n_states), 1:(control_system.n_controls), 1:T])
        @constraint(model, [t = 1:T], dfdu[:, :, t] .== [
            0 0
            0 0
            1 0
            0 1
        ])
    end

    # TODO: for now implement them by hand. Check later how much we would loose by doing AD.
    @constraint(model, dLdx[t = 2:T], dgdx[:, t] + λ[:, t - 1] - dfdx[:, :, t]' * λ[:, t] .== 0)
    @constraint(model, dLdu[t = 2:T], dgdu[:, t] - dfdu[:, :, t]' * λ[:, t] .== 0)

    # regularization
    @constraint(model, r' * r >= r_sqr_min)
    @constraint(model, r' * r + q' * q == 1)

    @objective(model, Min, inverse_objective(x; x̂))
    @time JuMP.optimize!(model)
    get_model_values(model, :q, :r, :x, :u, :λ), model, JuMP.value.(Q), JuMP.value.(R)
end

# The basis function for the cost model.
Q̃ = [diagm([1 // 3, 1 // 3, 0, 0]), diagm([0, 0, 2 // 3, 2 // 3])]
R̃ = [R]

inverse_solution, inverse_model, Q_est, R_est =
    solve_inverse_optimal_control(forward_solution.x, Q̃, R̃; control_system)

@testset "Solution Sanity" begin
    @test JuMP.termination_status(inverse_model) in (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL)
    @test Q_est[1, 1] / Q_est[4, 4] ≈ Q[1, 1] / Q[4, 4]
    @test Q_est[1, 1] / R_est[1, 1] ≈ Q[1, 1] / R[1, 1]
    @test Q_est[4, 4] / R_est[1, 1] ≈ Q[4, 4] / R[1, 1]
end
