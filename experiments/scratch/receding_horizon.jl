#using PartiallyObservedInverseGames.ForwardGame: ForwardGame
using GLMakie: GLMakie
using Makie: Makie

function visualize_trajectory_ineractive(trajectory; player_markers = ['●', '■'], color = "black")
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
T = 50
domain = 1.5

converged_gt, forward_solution_gt = ForwardGame.solve_game(
    IBRGameSolver(),
    control_system,
    player_cost_models_gt,
    x0,
    T;
    solver_attributes = (; print_level = 1),
)
gt_trajdata = TrajectoryVisualization.trajectory_data(control_system, forward_solution_gt.x)

visualize_trajectory_ineractive(gt_trajdata)
