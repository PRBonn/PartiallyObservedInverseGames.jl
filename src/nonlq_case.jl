using Test: @test, @testset
using UnPack: @unpack

# Optimization
using JuMP: JuMP, @NLconstraint, @constraint, @objective, @variable, @NLexpression, @expression
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
function add_unicycle_dynamics_constraints!(model, x, u)
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
    n_states, T = size(x)
    n_controls, _ = size(u)
    # TODO it's a bit ugly that we rely on these constraints to be present. We could check with
    # `haskey`.
    cosθ = model[:cosθ]
    sinθ = model[:sinθ]

    # jacobians of the dynamics in x
    @variable(model, dfdx[1:n_states, 1:n_states, 1:T])
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
    @expression(model, dfdu, [
        0 0
        0 0
        1 0
        0 1
    ] .* reshape(ones(T), 1, 1, :))

    (; dx = dfdx, du = dfdu)
end

symbol(s::Symbol) = s
symbol(s::JuMP.Containers.DenseAxisArrayKey) = only(s.I)

# TODO: [@DFK] Implement a "cost component library".
#
# Instructions: For every cost component:
#
# 1. Add a new weight to the `cost_model` NamedTuple below where the key is the component name and
# the value is the weight for the *true* true forward optimal control problem (the one that
# generates x̂)
#
# 2. Add a new key-value pair to the `g̃` NamedTuple in `add_forward_objective!`. The key is the same
# is for the `cost_model`.  The value computes thes *unweighted* cost for that component (e.g.
# collision cost).
#
# 3. Add the stage cost gradients to the NamedTuples `dg̃dx` and `dg̃du` in
# `add_forward_objective_gradients!`. Again, use the same keys as you used in the `cost_model`.
#
# Note: If you have non-quadratic/affine cost components, introduce an auxiliary variable +
# constraint (see e.g. the jacobian in line 51).

obstacle = (-0.5, 0.0) # Point to avoid.

function add_forward_objective!(model, x, u; weights)
    _, T = size(x)

    # Avoid a point. Assumes x = [px, py, ...]. Functional form is -log(|(x, y) - p|^2).
    @variable(model, sq_diff[t = 1:T])
    @constraint(model, [t = 1:T], sq_diff[t] == (x[1, t] - obstacle[1])^2 + (x[2, t] - obstacle[2])^2)
    @variable(model, prox_cost[t = 1:T])
    @NLconstraint(model, [t = 1:T], prox_cost[t] == -log(sq_diff[t] + 0.1))

    g̃ = (;
         goal = sum(x[:, t]' * x[:, t] for t in 1:T),
         control = sum(u[:, t]' * u[:, t] for t in 1:T),
         proximity = sum(prox_cost[t] for t in 1:T),
         )

    @objective(model, Min, sum(weights[k] * g̃[symbol(k)] for k in keys(weights)))
end

function add_forward_objective_gradients!(model, x, u; weights)
    # ∇ₓ of the proximity cost above.
    # ERROR(@lassepe): sq_diff key does not exist for some reason.
    @NLexpression(model, inv_sq_diff[t = 1:T], 1 / model[:sq_diff][t])
    @expression(model, dproxdx[t = 1:T], -(model[:x][1, t] - obstacle[1]) * inv_sq_diff[t])
    @expression(model, dproxdy[t = 1:T], -(model[:x][2, t] - obstacle[2]) * inv_sq_diff[t])

    dg̃dx = (;
            goal = 2 * x,
            control = zeros(size(x)),
            proximity = vcat(dproxdx', dproxdy', zeros(size(x, 1) - 2, T))
            )
    dgdx = sum(weights[k] * dg̃dx[symbol(k)] for k in keys(weights))

    dg̃du = (;
            goal = zeros(size(u)),
            control = 2 * u,
            proximity = zeros(size(u)),
            )
    dgdu = sum(weights[k] * dg̃du[symbol(k)] for k in keys(weights))

    (; dx = dgdx, du = dgdu)
end

control_system = (
    add_dynamics_constraints! = add_unicycle_dynamics_constraints!,
    add_dynamics_jacobians! = add_unicycle_dynamics_jacobians!,
    n_states = 4,
    n_controls = 2,
)
cost_model = (
    weights = (; goal = 1, control = 100, proximity = 0.5),
    add_objective! = add_forward_objective!,
    add_objective_gradients! = add_forward_objective_gradients!,
)
x0 = [1, 1, 0, 0]
T = 100

#====================================== forward optimal control ====================================#

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
    @unpack n_states, n_controls = control_system

    model = JuMP.Model(solver)
    set_solver_attributes!(model; silent, solver_attributes...)

    @variable(model, x[1:n_states, 1:T])
    @variable(model, u[1:n_controls, 1:T])
    control_system.add_dynamics_constraints!(model, x, u)
    @constraint(model, initial_condition, x[:, 1] .== x0)
    cost_model.add_objective!(model, x, u; cost_model.weights)
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
    x̂;
    control_system,
    cost_model,
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
    cmin = 1e-5,
)
    T = size(x̂)[2]
    @unpack n_states, n_controls = control_system

    model = JuMP.Model(solver)
    set_solver_attributes!(model; silent, solver_attributes...)

    # decision variable
    @variable(
        model,
        weights[keys(cost_model.weights)],
        start = 1 / sqrt(length(cost_model.weights))
    )
    @variable(model, x[1:n_states, 1:T])
    JuMP.set_start_value.(x, x̂)

    @variable(model, u[1:n_controls, 1:T])
    @variable(model, λ[1:n_states, 1:T]) # multipliers of the forward optimality condition

    # initial condition
    @constraint(model, initial_condition, x[:, 1] .== x̂[:, 1])
    control_system.add_dynamics_constraints!(model, x, u)
    # KKT conditions as constraints for forward optimality
    df = control_system.add_dynamics_jacobians!(model, x, u)
    dg = cost_model.add_objective_gradients!(model, x, u; weights)
    @constraint(
        model,
        dLdx[t = 2:T],
        dg.dx[:, t] + λ[:, t - 1] - (λ[:, t]' * df.dx[:, :, t])' .== 0
    )
    @constraint(model, dLdu[t = 2:T], dg.du[:, t] - (λ[:, t]' * df.du[:, :, t])' .== 0)
    # regularization
    # TODO: Think about what would be the right regularization now
    @constraint(model, weights .>= cmin)
    @constraint(model, weights' * weights == 1)
    @objective(model, Min, inverse_objective(x; x̂))

    @time JuMP.optimize!(model)
    get_model_values(model, :weights, :x, :u, :λ), model
end

inverse_solution, inverse_model =
    solve_inverse_optimal_control(forward_solution.x; control_system, cost_model)

@testset "Solution Sanity" begin
    @test JuMP.termination_status(inverse_model) in (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL)
    @test inverse_solution.weights[:goal] / inverse_solution.weights[:control] ≈
          cost_model.weights[:goal] / cost_model.weights[:control]
    @test isapprox(JuMP.objective_value(inverse_model), 0; atol = 1e-10)
end
