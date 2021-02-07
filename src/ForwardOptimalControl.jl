module ForwardOptimalControl

import ..DynamicsModelInterface
import ..JuMPUtils
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
    solver_attributes = (; print_level = 3),
)
    n_states, n_controls = size(only(unique(B)))
    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])
    @constraint(opt_model, linear_dynamics_constraints(x, u; A, B) .== 0)
    @constraint(opt_model, x[:, 1] .== x0)
    @objective(opt_model, Min, forward_quadratic_objective(x, u; Q, R))
    @time JuMP.optimize!(opt_model)
    JuMPUtils.get_values(; x, u), opt_model
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
    solver_attributes = (; print_level = 3),
    verbose = false,
)
    @unpack n_states, n_controls = control_system

    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # decision variables
    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])

    # initial guess
    JuMPUtils.init_if_hasproperty!(x, init, :x)
    JuMPUtils.init_if_hasproperty!(u, init, :u)

    # fix certain inputs
    for i in fixed_inputs
        JuMP.fix.(u[i, :], init.u[i, :])
    end

    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    @constraint(opt_model, x[:, 1] .== x0)
    cost_model.add_objective!(opt_model, x, u; cost_model.weights)
    time = @elapsed JuMP.optimize!(opt_model)
    verbose && @info time

    JuMPUtils.get_values(; x, u), opt_model
end

end
