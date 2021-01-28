module ForwardOptimalControl

import ..SolverUtils
import Ipopt
import JuMP

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export forward_quadratic_objective, linear_dynamics_constraints, solve_lqr, solve_optimal_control

#============================================= LQ case =============================================#

"The performance index for the forward optimal control problem."
function forward_quadratic_objective(x, u; Q, R)
    T = last(size(x))
    sum(x[:, t]' * Q * x[:, t] + u[:, t]' * R * u[:, t] for t in 1:T)
end

function linear_dynamics_constraints(x, u; A, B)
    reduce(hcat, ((x[:, t + 1] - A[t] * x[:, t] - B[t] * u[:, t]) for t in axes(x)[2][1:(end - 1)]))
end

"Solves a forward LQR problem using JuMP."
function solve_lqr(
    A,
    B,
    Q,
    R,
    x0,
    T;
    solver = Ipopt.Optimizer,
    silent = true,
    solver_attributes = (;)
)
    n_states, n_controls = size(only(unique(B)))
    model = JuMP.Model(solver)
    SolverUtils.set_solver_attributes!(model; silent, solver_attributes...)

    x = @variable(model, [1:n_states, 1:T])
    u = @variable(model, [1:n_controls, 1:T])
    @constraint(model, linear_dynamics_constraints(x, u; A, B) .== 0)
    @constraint(model, x[:, 1] .== x0)
    @objective(model, Min, forward_quadratic_objective(x, u; Q, R))
    @time JuMP.optimize!(model)
    SolverUtils.get_values(; x, u), model
end

#=========================================== Non-LQ-Case ===========================================#

"Solves a forward optimal control problem with protentially nonlinear dynamics and nonquadratic
costs using JuMP."
function solve_optimal_control(
    control_system,
    cost_model,
    x0,
    T;
    fixed_inputs = (),
    init = (),
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
)
    @unpack n_states, n_controls = control_system

    model = JuMP.Model(solver)
    SolverUtils.set_solver_attributes!(model; silent, solver_attributes...)

    # decision variables
    x = @variable(model, [1:n_states, 1:T])
    u = @variable(model, [1:n_controls, 1:T])

    # initial guess
    SolverUtils.init_if_hasproperty!(x, init, :x)
    SolverUtils.init_if_hasproperty!(u, init, :u)

    # fix certain inputs
    for i in fixed_inputs
        @constraint(model, u[fixed_inputs, :] .== init.u[fixed_inputs, :])
    end

    control_system.add_dynamics_constraints!(model, x, u)
    @constraint(model, x[:, 1] .== x0)
    cost_model.add_objective!(model, x, u; cost_model.weights)
    @time JuMP.optimize!(model)
    SolverUtils.get_values(; x, u), model
end

end
