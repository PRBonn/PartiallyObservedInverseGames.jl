module ForwardGame

import JuMP
import Ipopt

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack
using ..ForwardOptimalControl: solve_optimal_control
using ..SolverUtils

export solve_game

#================================ Iterated Best Open-Loop Response =================================#

struct IBRGameSolver end

# TODO: We could allow to pass an inner solver
function solve_game(
    ::IBRGameSolver,
    control_system,
    player_cost_models,
    x0,
    T;
    inner_solver_kwargs = (),
    max_ibr_rounds = 10,
    ibr_convergence_tolerance = 0.01,
)
    @unpack n_states, n_controls = control_system
    n_players = length(player_cost_models)

    last_ibr_solution = (; x = zeros(n_states, T), u = zeros(n_controls, T))
    last_player_solution = last_ibr_solution
    converged = false
    player_opt_models = resize!(JuMP.Model[], n_players)

    for i_ibr in 1:max_ibr_rounds
        for (player_idx, player_cost_model) in enumerate(player_cost_models)
            last_player_solution, player_opt_models[player_idx] = solve_optimal_control(
                control_system,
                player_cost_model,
                x0,
                T;
                fix_inputs = filter(i -> i ∉ player_cost_model.player_inputs, 1:n_controls),
                init = last_player_solution,
                inner_solver_kwargs...,
            )
        end

        converged =
            sum(x -> x^2, last_player_solution.x - last_ibr_solution.x) <= ibr_convergence_tolerance
        last_ibr_solution = last_player_solution

        if converged
            @info "Converged at ibr iterate: $i_ibr"
            break
        end
    end

    last_ibr_solution, converged, player_opt_models
end

#================================= Open-Loop KKT Nash Constraints ==================================#

struct KKTGameSolver end

# TODO handle missing "init" keys more gracefully (also in other solvers)
function solve_game(
    ::KKTGameSolver,
    control_system,
    player_cost_models,
    x0,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
    init = (),
)

    function init_if_hasproperty!(v, init, sym)
        if hasproperty(init, sym)
            JuMP.set_start_value.(v, getproperty(init, sym))
        end
    end

    n_players = 2
    @unpack n_states, n_controls = control_system
    model = JuMP.Model(solver)
    SolverUtils.set_solver_attributes!(model; silent, solver_attributes...)

    # Decision Variables
    # TODO: fix the variable access here.
    x = @variable(model, [1:n_states, 1:T])
    u = @variable(model, [1:n_controls, 1:T])
    λ = @variable(model, [1:n_states, 1:(T - 1), 1:n_players])

    # Initialization
    init_if_hasproperty!(λ, init, :λ)
    init_if_hasproperty!(x, init, :x)
    init_if_hasproperty!(u, init, :u)

    # constraints
    @constraint(model, x[:, 1] .== x0)
    control_system.add_dynamics_constraints!(model, x, u)
    df = control_system.add_dynamics_jacobians!(model, x, u)

    for (player_idx, cost_model) in enumerate(player_cost_models)
        @unpack player_inputs, weights = cost_model
        dJ = cost_model.objective_gradients(x, u; weights)

        # KKT Nash constraints
        @constraint(
            model,
            [t = 2:(T - 1)],
            dJ.dx[:, t] + λ[:, t - 1, player_idx] - (λ[:, t, player_idx]' * df.dx[:, :, t])' .== 0
        )
        @constraint(model, dJ.dx[:, T] + λ[:, T - 1, player_idx] .== 0)

        @constraint(
            model,
            [t = 1:(T - 1)],
            dJ.du[player_inputs, t] - (λ[:, t, player_idx]' * df.du[:, player_inputs, t])' .== 0
        )
        @constraint(model, dJ.du[player_inputs, T] .== 0)
    end

    @time JuMP.optimize!(model)
    SolverUtils.get_values(; x, u, λ), model
end

end
