using Test: @test, @testset

import Random

using JuMP: JuMP, @NLconstraint, @objective, @variable, @NLexpression
using SparseArrays: spzeros
using PartiallyObservedInverseGames.ForwardOptimalControl: solve_optimal_control
using PartiallyObservedInverseGames.InverseOptimalControl: solve_inverse_optimal_control
using PartiallyObservedInverseGames.TrajectoryVisualization: visualize_trajectory
using PartiallyObservedInverseGames.CostUtils: symbol

unique!(push!(LOAD_PATH, joinpath(@__DIR__, "utils")))
import TestUtils
import TestDynamics

#========================================== Cost Library ===========================================#

function register_shared_forward_cost_expressions!(opt_model, x, u; prox_min_regularization = 0.1)
    T = size(x, 2)
    @NLexpression(
        opt_model,
        regularized_sq_dist[t = 1:T],
        (x[1, t] - obstacle[1])^2 + (x[2, t] - obstacle[2])^2 + prox_min_regularization
    )
end

function add_forward_objective!(opt_model, x, u; weights)
    T = size(x, 2)
    register_shared_forward_cost_expressions!(opt_model, x, u)

    # Avoid a point. Assumes x = [px, py, ...]. Functional form is -log(|(x, y) - p|^2).
    @variable(opt_model, prox_cost[t = 1:T])
    @NLconstraint(opt_model, [t = 1:T], prox_cost[t] == -log(opt_model[:regularized_sq_dist][t]))

    g̃ = (;
        state_goal = sum(x[1:2, T_activate_goalcost:T] .^ 2),
        state_velocity = sum(x[3, :] .^ 2),
        state_proximity = sum(prox_cost),
        control_Δv = sum(u[1, :] .^ 2),
        control_Δθ = sum(u[2, :] .^ 2),
        control = sum(u .^ 2),
    )

    @objective(opt_model, Min, sum(weights[k] * g̃[symbol(k)] for k in keys(weights)))
end

function add_forward_objective_gradients!(opt_model, x, u; weights)
    n_states, T = size(x)
    n_controls = size(u, 1)
    register_shared_forward_cost_expressions!(opt_model, x, u)
    @variable(opt_model, dproxdx[1:T])
    @NLconstraint(
        opt_model,
        [t = 1:T],
        dproxdx[t] == -2 * (x[1, t] - obstacle[1]) / opt_model[:regularized_sq_dist][t]
    )
    @variable(opt_model, dproxdy[1:T])
    @NLconstraint(
        opt_model,
        [t = 1:T],
        dproxdy[t] == -2 * (x[2, t] - obstacle[2]) / opt_model[:regularized_sq_dist][t]
    )

    dJ̃dx = (;
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
    dJdx = sum(weights[k] * dJ̃dx[symbol(k)] for k in keys(weights))

    dJ̃du = (;
        state_goal = spzeros(n_controls, T),
        state_velocity = spzeros(n_controls, T),
        state_proximity = spzeros(n_controls, T),
        control_Δv = 2 * [u[1, :] zeros(T)]',
        control_Δθ = 2 * [spzeros(T) u[2, :]]',
        control = 2 * u,
    )
    dJdu = sum(weights[k] * dJ̃du[symbol(k)] for k in keys(weights))

    (; dx = dJdx, du = dJdu)
end

control_system = TestDynamics.Unicycle(0.1)

x0 = [-1, 1, 1, 0]
T = 100
obstacle = (-0.5, 0.25) # Point to avoid.
T_activate_goalcost = T
cost_model = (
    weights = (;
        state_goal = 100,
        state_velocity = 1,
        state_proximity = 1,
        control_Δv = 10,
        control_Δθ = 1,
    ),
    add_objective! = add_forward_objective!,
    add_objective_gradients! = add_forward_objective_gradients!,
)

#====================================== forward optimal control ====================================#

@testset "Forward Solution" begin
    forward_converged, forward_solution, forward_model =
        solve_optimal_control(control_system, cost_model, x0, T)
    @test forward_converged
    visualize_trajectory(control_system, forward_solution.x)
    global forward_solution
end

#===================================== Inverse Optimal Control =====================================#

observation_model = (; σ = 0.0, expected_observation = identity)

y = TestUtils.noisy_observation(observation_model, forward_solution.x)

inverse_converged, inverse_solution, inverse_model = solve_inverse_optimal_control(
    y;
    init = (; forward_solution.u),
    control_system,
    cost_model,
    observation_model,
)

#============================================== Tests ==============================================#

@testset "Inverse Solution" begin
    @test inverse_converged
    TestUtils.test_inverse_solution(inverse_solution.weights, cost_model.weights)
    TestUtils.test_data_fidelity(inverse_model, observation_model, forward_solution.x, y)
end
