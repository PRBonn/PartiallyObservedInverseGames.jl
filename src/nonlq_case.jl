using Test: @test, @testset

# Optimization
using JuMP: JuMP, @NLconstraint, @constraint, @objective, @variable, @expression
using LinearAlgebra: I, diagm
import Ipopt

# Visualization
import ElectronDisplay
import Plots
Plots.gr()
Plots.theme(:vibrant)

include("utils.jl")

#======================================== Global parameters ========================================#

# These constraints encode the dynamics of a unicycle with state layout x_t = [px, py, v, θ] and
# inputs u_t = [Δv, Δθ].
function add_unicycle_dynamics_constraints(model, x, u)
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
    # TODO it's a bit ugly that we rely on these constraints to be present. We could check with
    # `haskey`.
    cosθ = model[:cosθ]
    sinθ = model[:sinθ]

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

    (; dx = dfdx, du = dfdu)
end

function add_forward_objective!(model, x, u; Q, R)
    time_span = axes(x)[2]
    @expression(model, g_state, sum(x[:, t]' * Q * x[:, t] for t in time_span))
    @expression(model, g_control, sum(u[:, t]' * R * u[:, t] for t in time_span))
    @objective(model, Min, g_state + g_control)
end

function add_forward_objective_gradients!(model, x, u; Q, R)
    @expression(model, dgdx, 2 * Q * x)
    @expression(model, dgdu, 2 * R * u)
    (; dx = dgdx, du = dgdu)
end

control_system = (
    add_dynamics_constraints! = add_unicycle_dynamics_constraints,
    add_dynamics_jacobians! = add_unicycle_dynamics_jacobians!,
    n_states = 4,
    n_controls = 2,
)
cost_model = (
    Q = I,
    R = 100I,
    add_objective! = add_forward_objective!,
    add_objective_gradients! = add_forward_objective_gradients!,
)
x0 = [1, 1, 0, 0]
T = 100

#====================================== forward optimal control ====================================#

# TODO: think about handling of initial guees
"Solves a forward LQR problem using JuMP."
function solve_optimal_control(
    control_system,
    cost_model,
    x0,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
)
    model = JuMP.Model(solver)
    set_solver_attributes!(model; silent, solver_attributes...)

    @variable(model, x[1:(control_system.n_states), 1:T])
    @variable(model, u[1:(control_system.n_controls), 1:T])
    control_system.add_dynamics_constraints!(model, x, u)
    @constraint(model, initial_condition, x[:, 1] .== x0)
    cost_model.add_objective!(model, x, u; cost_model.Q, cost_model.R)
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

forward_solution, forward_model = solve_optimal_control(control_system, cost_model, x0, T)
visualize_unicycle_trajectory(forward_solution.x)

#===================================== Inverse Optimal Control =====================================#

function solve_inverse_optimal_control(
    x̂,
    Q̃,
    R̃;
    control_system,
    cost_model,
    r_sqr_min = 1e-5,
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
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
    control_system.add_dynamics_constraints!(model, x, u)

    # KKT conditions as constraints for forward optimality
    Q = sum(q .* Q̃)
    R = sum(r .* R̃)
    df = control_system.add_dynamics_jacobians!(model, x, u)
    dg = cost_model.add_objective_gradients!(model, x, u; Q, R)
    @constraint(model, dLdx[t = 2:T], dg.dx[:, t] + λ[:, t - 1] - (λ[:, t]' * df.dx[:, :, t])' .== 0)
    @constraint(model, dLdu[t = 2:T], dg.du[:, t] - (λ[:, t]' * df.du[:, :, t])' .== 0)

    # regularization
    @constraint(model, r' * r >= r_sqr_min)
    @constraint(model, r' * r + q' * q == 1)

    @objective(model, Min, inverse_objective(x; x̂))
    @time JuMP.optimize!(model)
    get_model_values(model, :q, :r, :x, :u, :λ), model, JuMP.value.(Q), JuMP.value.(R)
end

# The basis function for the cost model.
Q̃ = [diagm([1 // 3, 1 // 3, 0, 0]), diagm([0, 0, 2 // 3, 2 // 3])]
R̃ = [cost_model.R]

inverse_solution, inverse_model, Q_est, R_est =
    solve_inverse_optimal_control(forward_solution.x, Q̃, R̃; control_system, cost_model)

@testset "Solution Sanity" begin
    @test JuMP.termination_status(inverse_model) in (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL)
    @test Q_est[1, 1] / Q_est[4, 4] ≈ cost_model.Q[1, 1] / cost_model.Q[4, 4]
    @test Q_est[1, 1] / R_est[1, 1] ≈ cost_model.Q[1, 1] / cost_model.R[1, 1]
    @test Q_est[4, 4] / R_est[1, 1] ≈ cost_model.Q[4, 4] / cost_model.R[1, 1]
end
