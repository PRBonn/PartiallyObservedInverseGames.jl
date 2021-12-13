#using PartiallyObservedInverseGames.ForwardGame: ForwardGame
using GLMakie: GLMakie
using Makie: Makie
using Random: Random
using ProgressMeter: ProgressMeter

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

# Generate a longer demo
T_gt = 50
T_predict = 10
T_obs = 10
σ = 0.05
rng = Random.MersenneTwister(1)
observation_model = (; σ, estimator_setup_partial.expected_observation)

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

# TODO record a dataframe for analysis
# TODO warm-start with last solution (also costs weights)
ProgressMeter.@showprogress for t in T_obs:(T_gt - T_predict)
    obs_window = (t - T_obs + 1):t
    @assert length(obs_window) == T_obs
    y = observations[:, obs_window]
    solve_inverse_game(
        InverseKKTConstraintSolver(),
        y;
        control_system,
        observation_model,
        player_cost_models = player_cost_models_gt,
        T_predict,
        solver_attributes = (; print_level = 1),
        estimator_setup_partial.cmin,
        estimator_setup_partial.pre_solve_kwargs,
    )
end

#gt_trajdata = TrajectoryVisualization.trajectory_data(control_system, forward_solution_gt.x)
#
#visualize_trajectory_ineractive(gt_trajdata)
