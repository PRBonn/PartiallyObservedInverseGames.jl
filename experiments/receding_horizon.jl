module RecedingHorizon
const project_root_dir = realpath(joinpath(@__DIR__, ".."))

#using PartiallyObservedInverseGames.ForwardGame: ForwardGame
using Random: Random
using ProgressMeter: ProgressMeter
using GLMakie: GLMakie
using Makie: Makie

include("utils/preamble.jl")
include("utils/unicycle_online_setup.jl")

function visualize_trajectory_ineractive(
    trajectory;
    player_markers = ['●', '■'],
    color = "black",
    domain = 1.5,
)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; aspect = 1, limits = ((-domain, domain), (-domain, domain)))

    time_slider = Makie.Slider(fig[2, 1]; range = 1:maximum(s.t for s in trajectory))

    players = unique(s.player for s in trajectory)
    for (ii, p) in pairs(players)
        positions = [Makie.Point2f(s.px, s.py) for s in trajectory if s.player == p]
        marker = player_markers[ii]
        Makie.lines!(ax, positions; color)
        Makie.scatter!(ax, Makie.@lift(positions[$(time_slider.value)]); marker, color)
    end

    fig
end

function visualize_receding_horizon(sim_steps, position_indices; domain = 1.5)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; aspect = 1, limits = ((-domain, domain), (-domain, domain)))

    time_slider = Makie.Slider(fig[2, 1]; range = 1:maximum(s.t for s in sim_steps))

    step = Makie.@lift sim_steps[$(time_slider.value)]

    for (ii, p) in pairs(position_indices)
        positions = Makie.@lift [Makie.Point2f(x) for x in eachcol($step.x[p, :])]
        Makie.lines!(ax, positions)
        Makie.scatter!(ax, Makie.@lift $positions[$step.T_obs])
    end

    fig
end

function simulate(; T_gt = 50, T_predict = 10, T_obs = 10, σ = 0.05, t_start = 2)

    # Generate a longer demo
    rng = Random.MersenneTwister(1)
    observation_model = (; σ, expected_observation = identity)

    converged_gt, forward_solution_gt = ForwardGame.solve_game(
        IBRGameSolver(),
        control_system,
        player_cost_models_gt,
        x0,
        T_gt;
        solver_attributes = (; print_level = 1),
    )

    observations = observation_model.expected_observation(
        forward_solution_gt.x .+ σ * randn(rng, size(forward_solution_gt.x)),
    )

    # TODO warm-start with last solution (also costs weights)
    sim_steps = NamedTuple[]

    ProgressMeter.@showprogress for t in t_start:(T_gt - T_predict)
        obs_window = max((t - T_obs + 1), 1):t
        @assert length(obs_window) <= T_obs
        y = observations[:, obs_window]
        converged, solution = solve_inverse_game(
            InverseKKTConstraintSolver(),
            y;
            control_system,
            observation_model,
            player_cost_models = player_cost_models_gt,
            T_predict,
            solver_attributes = (; print_level = 1),
            cmin = 1.e-3,
            pre_solve_kwargs = (; u_regularization = 1e-5),
        )

        @assert converged

        push!(sim_steps, (; t, T_obs = length(obs_window), solution...))
    end

    #gt_trajdata = TrajectoryVisualization.trajectory_data(control_system, forward_solution_gt.x)
    #
    #visualize_trajectory_ineractive(gt_trajdata)
    sim_steps
end
end
