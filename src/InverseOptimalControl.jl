module InverseOptimalControl

import ..SolverUtils
import Ipopt
import JuMP

using ..ForwardOptimalControl: linear_dynamics_constraints, forward_quadratic_objective
using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export solve_inverse_optimal_control

#============================================= LQ case =============================================#

"The lagrangian of the forward LQR problem."
function lqr_lagrangian(x, u, λ; Q, R, A, B)
    c_dyn = linear_dynamics_constraints(x, u; A, B)
    @views forward_quadratic_objective(x, u; Q, R) +
           sum(λ[:, t]' * c_dyn[:, t] for t in axes(x)[2][1:(end - 1)])
end

"The hand-written gradient of the forward LQR problem in x."
function lqr_lagrangian_grad_x(x, u, λ; Q, R, A, B)
    T = size(x, 2)
    hcat(
        2 * Q * x[:, 1] - A[1]' * λ[:, 1], # special handling of first time step
        reduce(hcat, (2 * Q * x[:, t] + λ[:, t - 1] - A[t]' * λ[:, t] for t in 2:(T - 1))),
        2 * Q * x[:, T] + λ[:, T - 1],
    )
end

"The hand-written gradient of the forward LQR problem in u."
function lqr_lagrangian_grad_u(x, u, λ; Q, R, A, B)
    T = size(x, 2)
    hcat(reduce(hcat, (2 * R * u[:, t] - B[t]' * λ[:, t] for t in 1:(T - 1))), 2 * R * u[:, T])
end

"""
Solves aninverse LQR problem using JuMP.

`x̂` is the (observed) state trajectory for which we wish to find the cost parametrization.

`Q̃` and `R̃` are iterables of quadatic state and control cost matrices for which the weight
vectors`q` and `r` are to be estimated.
"""
function solve_inverse_lqr(
    x̂,
    Q̃,
    R̃;
    A,
    B,
    r_sqr_min = 1e-5,
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
)
    T = size(x̂)[2]
    n_states, n_controls = size(only(unique(B)))
    model = JuMP.Model(solver)
    SolverUtils.set_solver_attributes!(model; silent, solver_attributes...)

    # decision variable
    q = @variable(model, [1:length(Q̃)], lower_bound = 0)
    r = @variable(model, [1:length(R̃)], lower_bound = 0)
    x = @variable(model, [1:n_states, 1:T])
    u = @variable(model, [1:n_controls, 1:T])
    λ = @variable(model, [1:n_states, 1:(T - 1)]) # multipliers of the forward LQR condition

    # initial condition
    @constraint(model, x[:, 1] .== x̂[:, 1])
    # dynamics
    @constraint(model, linear_dynamics_constraints(x, u; A, B) .== 0)
    # Optimality conditions (KKT) of forward LQR show up as a constraints
    Q = sum(q .* Q̃)
    R = sum(r .* R̃)
    ∇ₓL = @constraint(model, lqr_lagrangian_grad_x(x, u, λ; Q, R, A, B)[:, 2:end] .== 0)
    ∇ᵤL = @constraint(model, lqr_lagrangian_grad_u(x, u, λ; Q, R, A, B) .== 0)
    # regularization
    @constraint(model, r' * r >= r_sqr_min)
    @constraint(model, r' * r + q' * q == 1)

    @objective(model, Min, sum((x .- x̂) .^ 2))
    @time JuMP.optimize!(model)
    SolverUtils.get_values(; q, r, x, u, λ, ∇ₓL, ∇ᵤL), model, JuMP.value.(Q), JuMP.value.(R)
end

#=========================================== Non-LQ-Case ===========================================#

function solve_inverse_optimal_control(
    y;
    control_system,
    cost_model,
    observation_model,
    fixed_inputs = (),
    init = (),
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
    cmin = 1e-5,
)
    T = size(y)[2]
    @unpack n_states, n_controls = control_system

    model = JuMP.Model(solver)
    SolverUtils.set_solver_attributes!(model; silent, solver_attributes...)

    # decision variable
    weights = @variable(model, [keys(cost_model.weights)],)
    x = @variable(model, [1:n_states, 1:T])
    u = @variable(model, [1:n_controls, 1:T])
    λ = @variable(model, [1:n_states, 1:(T - 1)]) # multipliers of the forward optimality condition

    # initialization
    if hasproperty(init, :weights) && !isnothing(init.weights)
        for k in keys(init.weights)
            JuMP.set_start_value(weights[k], init.weights[k])
        end
    else
        JuMP.set_start_value.(weights, 1 / length(weights))
    end

    # SolverUtils.init_if_hasproperty!(weights, init, :weights, default = 1 / length(weights))
    # TODO: This is not always correct. Technically we would want to use an inverse observation
    # model here if it exists (mapping from observationi to state components)
    JuMP.set_start_value.(x[CartesianIndices(y)], y)
    SolverUtils.init_if_hasproperty!(u, init, :u)

    # constraints
    if iszero(observation_model.σ)
        @constraint(model, observation_model.expected_observation(x[:, 1]) .== y[:, 1])
    end
    control_system.add_dynamics_constraints!(model, x, u)
    # figure out which inputs we control and fix all others
    controlled_inputs = filter(i -> i ∉ fixed_inputs, 1:n_controls)
    for i in fixed_inputs
        @constraint(model, u[fixed_inputs, :] .== init.u[fixed_inputs, :])
    end
    # Require forward-optimality for all *controlled* inputs.
    df = control_system.add_dynamics_jacobians!(model, x, u)
    dJ = cost_model.add_objective_gradients!(model, x, u; weights)
    @constraint(
        model,
        [t = 2:(T - 1)],
        dJ.dx[:, t] + λ[:, t - 1] - (λ[:, t]' * df.dx[:, :, t])' .== 0
    )
    @constraint(model, dJ.dx[:, T] + λ[:, T - 1] .== 0)
    @constraint(
        model,
        [t = 1:(T - 1)],
        dJ.du[controlled_inputs, t] - (λ[:, t]' * df.du[:, controlled_inputs, t])' .== 0
    )
    @constraint(model, dJ.du[controlled_inputs, T] .== 0)
    # regularization
    # TODO: There might be a smarter regularization here. Rather, we want there to be non-zero cost
    # for all inputs (i.e. enforce positive definiteness of the input cost.)
    @constraint(model, weights .>= cmin)
    @constraint(model, sum(weights) == 1)

    # The inverse objective: match the observed demonstration
    @objective(model, Min, sum((observation_model.expected_observation(x) .- y) .^ 2))

    @time JuMP.optimize!(model)
    SolverUtils.get_values(; weights, x, u, λ), model
end

end
