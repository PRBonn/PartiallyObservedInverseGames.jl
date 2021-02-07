module ForwardGame

import JuMP
import Ipopt
import ..DynamicsModelInterface
import ..JuMPUtils
import ..ForwardOptimalControl

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export IBRGameSolver, KKTGameSolver, solve_game

#================================ Iterated Best Open-Loop Response =================================#

struct IBRGameSolver end

# TODO: We could allow to pass an inner solver
function solve_game(
    ::IBRGameSolver,
    control_system,
    player_cost_models,
    x0,
    T;
    init = (; x = zeros(control_system.n_states, T), u = zeros(control_system.n_controls, T)),
    max_ibr_rounds = 10,
    ibr_convergence_tolerance = 0.01,
    verbose = false,
    inner_solver_kwargs...,
)
    @unpack n_states, n_controls = control_system
    n_players = length(player_cost_models)

    last_ibr_solution = init
    last_player_solution = last_ibr_solution
    player_opt_models = resize!(JuMP.Model[], n_players)
    converged = false

    for i_ibr in 1:max_ibr_rounds
        for (player_idx, player_cost_model) in enumerate(player_cost_models)
            last_player_solution, player_opt_models[player_idx] =
                ForwardOptimalControl.solve_optimal_control(
                    control_system,
                    player_cost_model,
                    x0,
                    T;
                    fixed_inputs = filter(i -> i ∉ player_cost_model.player_inputs, 1:n_controls),
                    init = last_player_solution,
                    inner_solver_kwargs...,
                    verbose,
                )
        end

        converged =
            sum(Δu -> Δu^2, last_player_solution.u - last_ibr_solution.u) <=
            ibr_convergence_tolerance
        last_ibr_solution = last_player_solution

        if converged
            verbose && @info "Converged at ibr iterate: $i_ibr"
            break
        end
    end

    converged || @warn "IBR terminated pre-maturely."

    converged, last_ibr_solution, player_opt_models
end

#================================= Open-Loop KKT Nash Constraints ==================================#

struct KKTGameSolver end

function solve_game(
    ::KKTGameSolver,
    control_system,
    player_cost_models,
    x0,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (; print_level = 3),
    init = (),
    verbose = false,
)

    n_players = length(player_cost_models)
    @unpack n_states, n_controls = control_system

    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # Decision Variables
    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])
    λ = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])

    # Initialization
    JuMPUtils.init_if_hasproperty!(λ, init, :λ)
    JuMPUtils.init_if_hasproperty!(x, init, :x)
    JuMPUtils.init_if_hasproperty!(u, init, :u)

    # constraints
    @constraint(opt_model, x[:, 1] .== x0)
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

    for (player_idx, cost_model) in enumerate(player_cost_models)
        @unpack player_inputs, weights = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

        # KKT Nash constraints
        @constraint(
            opt_model,
            [t = 2:(T - 1)],
            dJ.dx[:, t] + λ[:, t - 1, player_idx] - (λ[:, t, player_idx]' * df.dx[:, :, t])' .== 0
        )
        @constraint(opt_model, dJ.dx[:, T] + λ[:, T - 1, player_idx] .== 0)

        @constraint(
            opt_model,
            [t = 1:(T - 1)],
            dJ.du[player_inputs, t] - (λ[:, t, player_idx]' * df.du[:, player_inputs, t])' .== 0
        )
        @constraint(opt_model, dJ.du[player_inputs, T] .== 0)
    end

    time = @elapsed JuMP.optimize!(opt_model)
    verbose && @info time

    JuMPUtils.isconverged(opt_model), JuMPUtils.get_values(; x, u, λ), opt_model
end

end
