using Test: @test, @testset
using UnPack: @unpack

# Optimization
using JuMP: JuMP, @NLconstraint, @constraint, @objective, @variable, @NLexpression, @expression
using LinearAlgebra: I, diagm
using SparseArrays: spzeros
import Random
import Ipopt

# Visualization
import ElectronDisplay
import Plots
Plots.gr()
Plots.theme(:vibrant)

include("utils.jl")
include("unicycle.jl")

#======================================== Global parameters ========================================#


symbol(s::Symbol) = s
symbol(s::JuMP.Containers.DenseAxisArrayKey) = only(s.I)

function register_shared_forward_cost_expressions!(model, x, u; prox_min = 0.1)
    T = size(x, 2)
    @NLexpression(
        model,
        regularized_sq_dist[t = 1:T],
        (x[1, t] - obstacle[1])^2 + (x[2, t] - obstacle[2])^2 + prox_min
    )
end

function add_forward_objective!(model, x, u; weights)
    T = size(x, 2)
    register_shared_forward_cost_expressions!(model, x, u)

    # Avoid a point. Assumes x = [px, py, ...]. Functional form is -log(|(x, y) - p|^2).
    @variable(model, prox_cost[t = 1:T])
    @NLconstraint(model, [t = 1:T], prox_cost[t] == -log(model[:regularized_sq_dist][t]))

    g̃ = (;
        state_goal = sum(x[1:2, T_activate_goalcost:T] .^ 2),
        state_velocity = sum(x[3, :] .^ 2),
        state_proximity = sum(prox_cost),
        control_Δv = sum(u[1, :] .^ 2),
        control_Δθ = sum(u[2, :] .^ 2),
        control = sum(u .^ 2),
    )

    @objective(model, Min, sum(weights[k] * g̃[symbol(k)] for k in keys(weights)))
end

function add_forward_objective_gradients!(model, x, u; weights)
    n_states, T = size(x)
    n_controls = size(u, 1)
    register_shared_forward_cost_expressions!(model, x, u)
    @variable(model, dproxdx[1:T])
    @NLconstraint(
        model,
        [t = 1:T],
        dproxdx[t] == -2 * (x[1, t] - obstacle[1]) / model[:regularized_sq_dist][t]
    )
    @variable(model, dproxdy[1:T])
    @NLconstraint(
        model,
        [t = 1:T],
        dproxdy[t] == -2 * (x[2, t] - obstacle[2]) / model[:regularized_sq_dist][t]
    )

    dg̃dx = (;
        state_goal = 2 * [
            zeros(2, T_activate_goalcost - 1) x[1:2, T_activate_goalcost:T]
            zeros(n_states - 2, T)
        ],
        state_velocity = 2 * [spzeros(T, 2) x[3, :] spzeros(T)]',
        state_proximity = [dproxdx dproxdy spzeros(T, n_states - 2)]',
        control_Δv = spzeros(n_states, T),
        control_Δθ = spzeros(n_states, T),
        control = spzeros(n_states, T),
    )
    dgdx = sum(weights[k] * dg̃dx[symbol(k)] for k in keys(weights))

    dg̃du = (;
        state_goal = spzeros(n_controls, T),
        state_velocity = spzeros(n_controls, T),
        state_proximity = spzeros(n_controls, T),
        control_Δv = 2 * [u[1, :] zeros(T)]',
        control_Δθ = 2 * [spzeros(T) u[2, :]]',
        control = 2 * u,
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

# TODO: maybe limit the region of proximity cost
const x0 = [-1, 1, 0, 0]
const T = 100
const obstacle = (-0.5, 0.25) # Point to avoid.
const T_activate_goalcost = 50
cost_model = (
    weights = (;
        state_goal = 100,
        state_velocity = 1_000, # TODO: have a higher cost for driving backwards
        state_proximity = 1,
        control_Δv = 100,
        control_Δθ = 10, # TODO: Delay orientation input once more
        # control = 100,
    ),
    add_objective! = add_forward_objective!,
    add_objective_gradients! = add_forward_objective_gradients!,
)

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

forward_solution, forward_model = solve_optimal_control(control_system, cost_model, x0, T)
visualize_unicycle_trajectory(forward_solution.x)

#===================================== Inverse Optimal Control =====================================#

function solve_inverse_optimal_control(
    y,
    û = nothing;
    control_system,
    cost_model,
    observation_model,
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
    cmin = 1e-5,
    max_trajectory_error = nothing,
)
    T = size(y)[2]
    @unpack n_states, n_controls = control_system

    model = JuMP.Model(solver)
    set_solver_attributes!(model; silent, solver_attributes...)

    # decision variable
    @variable(model, weights[keys(cost_model.weights)],)
    @variable(model, x[1:n_states, 1:T])
    @variable(model, u[1:n_controls, 1:T])
    @variable(model, λ[1:n_states, 1:T]) # multipliers of the forward optimality condition
    # TODO: Are there smarter initial guesses that we can make for `u` and `λ`?
    JuMP.set_start_value.(weights, 1 / length(cost_model.weights))
    JuMP.set_start_value.(x[CartesianIndices(y)], y)
    if !isnothing(û)
        JuMP.set_start_value.(u[CartesianIndices(û)], û)
    end

    # constraints
    if !isnothing(max_trajectory_error)
        @constraint(model, sum(x .- y) .^ 2 <= max_trajectory_error)
    end
    if iszero(observation_model.σ)
        @constraint(model, observation_model.expected_observation(x[:, 1]) .== y[:, 1])
    end
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
    # TODO: There might be a smarter regularization here. Rather, we want there to be non-zero cost
    # for all inputs.
    @constraint(model, weights .>= cmin)
    @constraint(model, sum(weights) == 1)

    # The inverse objective: match the observed demonstration
    @objective(model, Min, sum((observation_model.expected_observation(x) .- y) .^ 2))

    @time JuMP.optimize!(model)
    get_model_values(model, :weights, :x, :u, :λ), model
end

observation_model = (; σ = 0.0, expected_observation = identity)

y = let
    ŷ = observation_model.expected_observation(forward_solution.x)
    ŷ + randn(Random.MersenneTwister(1), size(ŷ)) .* observation_model.σ
end

inverse_solution, inverse_model = solve_inverse_optimal_control(
    y,
    forward_solution.u;
    control_system,
    cost_model,
    observation_model,
)

#============================================== Tests ==============================================#

@testset "Solution Sanity" begin
    @test JuMP.termination_status(inverse_model) in (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL)
    atol = 1e-2

    w_total_inverse = sum(inverse_solution.weights)
    w_total_forward = sum(cost_model.weights)

    for k in keys(cost_model.weights)
        @info k
        @test isapprox(
            inverse_solution.weights[k] / w_total_inverse,
            cost_model.weights[k] / w_total_forward;
            atol = atol,
        )
    end

    ŷ = observation_model.expected_observation(forward_solution.x)
    ê_sq = sum((ŷ .- y) .^ 2)
    @test JuMP.objective_value(inverse_model) <= ê_sq + 1e-2
end
