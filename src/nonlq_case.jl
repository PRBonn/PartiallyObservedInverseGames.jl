# Optimization
using JuMP: JuMP, @NLconstraint, @constraint, @objective, @variable
using LinearAlgebra: I
import Ipopt

# Visualization
import ElectronDisplay
import Plots
Plots.gr()
Plots.theme(:vibrant)

include("utils.jl")

#======================================== Global parameters ========================================#

control_system =
    (dynamics_constraints! = add_unicycle_dynamics_constraints!, n_states = 4, n_controls = 2)
Q = I
R = 100I
x0 = [1, 1, 0, 0]
T = 100

# TODO: generalize this to take arbitrary nonlinear, vector-valued functions NOTE: This seems to be
# non-trivial to generalize; This would require some meta programming macromagic or generated
# function perhaps.
#
# These constraints encode the dynamics of a unicycle with state layout x_t = [px, py, v, θ] and
# inputs u_t = [Δv, Δθ].
function add_unicycle_dynamics_constraints!(model, x, u, T)
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
    # Δv
    @constraint(model, dynamics_v[t = 1:(T - 1)], x[3, t + 1] .== x[3, t] .+ u[1, t])
    # Δθ
    @constraint(model, dynamics_θ[t = 1:(T - 1)], x[4, t + 1] .== x[4, t] .+ u[2, t])
end

#====================================== forward optimal control ====================================#

# TODO: think about handling of initial guees
"Solves a forward LQR problem using JuMP."
function solve_optimal_control(control_system, Q, R, x0, T)
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, x[1:(control_system.n_states), 1:T])
    @variable(model, u[1:(control_system.n_controls), 1:T])
    control_system.dynamics_constraints!(model, x, u, T)
    @constraint(model, initial_condition, x[:, 1] .== x0)
    @objective(model, Min, forward_objective(x, u; Q, R))
    JuMP.optimize!(model)
    get_model_values(model, :x, :u), model
end

# unicycle_solution, forward_model = let
#     solve_optimal_control(control_system, Q, R, x0, T)
# end

function visualize_unicycle_trajectory(x)
    unicycle_viz = Plots.plot(
        x[1, :],
        x[2, :],
        quiver = (x[3, :] .* cos.(x[4, :]), x[3, :] .* sin.(x[4, :])),
        st = :quiver,
    )
end

# visualize_unicycle_trajectory(unicycle_solution.x)
# The basis function for the cost model.

#===================================== Inverse Optimal Control =====================================#

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

function solve_inverse_optimal_control(x̂, Q̃, R̃; control_system, r_sqr_min = 1e-5)
    T = size(x̂)[2]
    model = JuMP.Model(Ipopt.Optimizer)

    # decision variable
    @variable(model, q[1:length(Q̃)] >= 0)
    @variable(model, r[1:length(R̃)] >= 0)
    @variable(model, x[1:(control_system.n_states), 1:T])
    @variable(model, u[1:(control_system.n_controls), 1:T])
    @variable(model, λ[1:(control_system.n_states), 1:T]) # multipliers of the forward optimality condition

    # initial condition
    @constraint(model, initial_condition, x[:, 1] .== x̂[:, 1])
    control_system.dynamics_constraints!(model, x, u)

    # Optimality conditions (KKT) of forward LQR show up as a constraints
    Q = sum(q .* Q̃)
    R = sum(r .* R̃)

    # TODO: think about these contraints; What do they need to be in the nonlinear case?
    # @constraint(model, ∇ₓL, lqr_lagrangian_grad_x(x, u, λ; Q, R, A, B, T)[:, 2:end] .== 0)
    # @constraint(model, ∇ᵤL, lqr_lagrangian_grad_u(x, u, λ; Q, R, A, B, T) .== 0)
    begin

        @NLconstraint(model, dLdpx[t = 1:(T - 1)], )
        #@NLconstraint(model, dLdpy[t = 1:(T - 1)], )
        #@NLconstraint(model, dLdv[t = 1:(T - 1)], )
        #@NLconstraint(model, dLdθ[t = 1:(T - 1)], )
        #@NLconstraint(model, dLdΔv[t = 1:(T - 1)], )
        #@NLconstraint(model, dLdΔθ[t = 1:(T - 1)], )
    end
    # regularization
    @constraint(model, r' * r >= r_sqr_min)
    @constraint(model, r' * r + q' * q == 1)

    @objective(model, Min, inverse_objective(x; x̂))
    JuMP.optimize!(model)
    get_model_values(model, :q, :r, :x, :u, :λ), model, JuMP.value.(Q), JuMP.value.(R)
end
