module RecedingHorizon
const project_root_dir = realpath(joinpath(@__DIR__, ".."))

#using PartiallyObservedInverseGames.ForwardGame: ForwardGame
using Random: Random
using ProgressMeter: ProgressMeter
using CairoMakie: CairoMakie
using Makie: Makie

include("utils/preamble.jl")
load_cache_if_not_defined!("receding_horizon")

function setup_highway(player_configurations; Δt = 1.0, T)
    x0 = mapreduce(vcat, player_configurations) do player_config
        [
            player_config.initial_lane,
            player_config.initial_progress,
            player_config.initial_speed,
            deg2rad(90),
        ]
    end

    control_system = TestDynamics.ProductSystem([
        TestDynamics.Unicycle(Δt),
        TestDynamics.Unicycle(Δt),
        TestDynamics.Unicycle(Δt),
        TestDynamics.Unicycle(Δt),
        TestDynamics.Unicycle(Δt),
    ])

    position_indices = mapreduce(vcat, eachindex(control_system.subsystems)) do subsystem_idx
        TestDynamics.state_indices(control_system, subsystem_idx)[1:2]
    end

    partial_state_indices = mapreduce(vcat, eachindex(control_system.subsystems)) do subsystem_idx
        TestDynamics.state_indices(control_system, subsystem_idx)[[1, 2, 4]]
    end

    player_cost_models_gt = map(Iterators.countfrom(1), player_configurations) do ii, player_config
        cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model(;
            player_idx = ii,
            control_system,
            T,
            goal_position = nothing,
            weights = (;
                state_proximity = player_config.prox_cost,
                state_velocity = player_config.speed_cost,
                control_Δv = 1,
                control_Δθ = 1,
            ),
            fix_costs = (; state_goal = 100, state_lane = 0.1, state_orientation = 2.0),
            y_lane_center = player_config.target_lane,
            target_speed = player_config.target_speed,
        )
    end

    (; T, x0, control_system, position_indices, partial_state_indices, player_cost_models_gt)
end

function _visualize_receding_horizon_frame!(
    ax,
    t::Makie.Observable,
    sim_result;
    subsampling = 2,
    temporal_markersize = 3,
)
    step = Makie.@lift sim_result.estimator_steps[$t]

    for p in sim_result.position_indices
        buffer_size = Makie.@lift size($step.y_full, 2)

        ground_truth = [Makie.Point2f(x[p]) for x in eachcol(sim_result.forward_solution_gt.x)]
        prediction = Makie.@lift [Makie.Point2f(x[p]) for x in eachcol($step.x)]
        observations_full =
            Makie.@lift [Makie.Point2f(y_full[p]) for y_full in eachcol($step.y_full)]

        # line and point for ground truth
        prediction_window = Makie.@lift ($t):subsampling:($t + 15)
        history_window = Makie.@lift unique!([1:subsampling:($t); $t])
        ground_truth_prediction = Makie.@lift ground_truth[$prediction_window]
        ground_truth_history = Makie.@lift ground_truth[$history_window]

        Makie.lines!(ax, ground_truth_history; color = ("black", 0.2))
        Makie.scatter!(
            ax,
            ground_truth_history;
            color = ("black", 0.2),
            markersize = temporal_markersize,
        )

        Makie.lines!(ax, ground_truth_prediction; color = "black")
        Makie.scatter!(
            ax,
            ground_truth_prediction;
            color = "black",
            markersize = temporal_markersize,
        )
        Makie.scatter!(ax, Makie.@lift ground_truth[$step.t]; color = "black")

        # line and point for prediction
        Makie.lines!(ax, prediction; color = "royalblue")
        Makie.scatter!(ax, Makie.@lift $prediction[$buffer_size]; color = "royalblue")

        # raw observations in the buffer
        Makie.scatter!(ax, observations_full; color = ("gray", 0.5))
    end

    Makie.rotate!(ax.scene, -pi / 2)
    ax.aspect = Makie.DataAspect()
    ax
end

function visualize_receding_horizon(
    sim_result,
    time_steps::Colon = :;
    resolution = (600, 300),
    limits = ((-5, 32), (-2, 2)),
)
    fig = Makie.Figure(; resolution)
    ax = Makie.Axis(fig[1, 1]; limits)
    time_slider = Makie.Slider(fig[2, 1]; range = eachindex(sim_result.estimator_steps))
    _visualize_receding_horizon_frame!(ax, time_slider.value, sim_result)
    fig
end

function visualize_receding_horizon(
    sim_result,
    time_steps;
    limits = ((-5, 25), (-2, 2)),
    resolution = (600, length(time_steps) * 135),
)
    axislabelfont = "Noto-Bold"

    fig = Makie.Figure(; resolution)
    ax_main = fig[1:length(time_steps), 1] = Makie.GridLayout()

    axes = map(enumerate(time_steps)) do (k, t)
        ax = Makie.Axis(
            ax_main[k, 1];
            limits,
            title = "t = $(t - 1) [s]",
            titlefont = axislabelfont,
            titlegap = 2,
            titlealign = :left,
        )
        time = Makie.Observable(t)
        _visualize_receding_horizon_frame!(ax, time, sim_result)
    end

    Makie.rowgap!(ax_main, 5)

    axes[end].xlabel = "Position x [m]"
    axes[end].xlabelfont = axislabelfont
    axes[end].bottomspinevisible = true
    axes[3].ylabel = "Position y [m]"
    axes[3].ylabelfont = axislabelfont

    fig
end

function simulate(
    setup;
    T_predict = 10,
    T_obs = 5,
    σ = 0.05,
    t_start = 5,
    u_regularization = 1e-5,
    prior_weight = 1e-2,
    use_heuristic_prior = true,
)

    # Generate a longer demo
    rng = Random.MersenneTwister(1)
    observation_model = (; σ, expected_observation = x -> x[setup.partial_state_indices, :])

    converged_gt, forward_solution_gt = ForwardGame.solve_game(
        IBRGameSolver(),
        setup.control_system,
        setup.player_cost_models_gt,
        setup.x0,
        setup.T;
        solver_attributes = (; print_level = 1),
    )

    observations_full = forward_solution_gt.x .+ σ * randn(rng, size(forward_solution_gt.x))
    observations_partial = observation_model.expected_observation(observations_full)

    # TODO warm-start with last solution (also costs weights)
    estimator_steps = NamedTuple[]
    init = nothing
    previous_solution = nothing
    ProgressMeter.@showprogress for t in t_start:(setup.T - T_predict)
        obs_window = max((t - T_obs + 1), 1):t
        @assert length(obs_window) <= T_obs
        y = observations_partial[:, obs_window]

        # if we have a previous solution we can use it for warmstarting
        if !isnothing(previous_solution)
            init = let
                presolve_converged, presolve_solution = InversePreSolve.pre_solve(
                    y,
                    nothing;
                    setup.control_system,
                    observation_model,
                    #observation_model = (; expected_observation = identity),
                    T = length(obs_window) + T_predict,
                    u_regularization,
                    solver_attributes = (; print_level = 1),
                )
                @assert presolve_converged

                (; presolve_solution..., player_weights = previous_solution.player_weights)
            end
        end

        prior = if isnothing(previous_solution) || !use_heuristic_prior
            nothing
        else
            (;
                previous_solution.player_weights,
                x = previous_solution.x[:, (1:length(obs_window)) .+ 1],
                u = previous_solution.u[:, (1:length(obs_window)) .+ 1],
            )
        end

        converged, solution = solve_inverse_game(
            InverseKKTConstraintSolver(),
            y;
            setup.control_system,
            observation_model,
            player_cost_models = setup.player_cost_models_gt,
            T_predict,
            solver_attributes = (; print_level = 1),
            cmin = 1.e-3,
            pre_solve = true,
            prior,
            prior_weight,
            init_with_observation = false,
            pre_solve_kwargs = (; u_regularization),
            init,
        )

        @assert converged
        previous_solution = solution

        push!(estimator_steps, (; t, y_full = observations_full[:, obs_window], solution...))
    end

    (;
        estimator_steps,
        forward_solution_gt,
        position_indices = setup.position_indices |> ii -> Iterators.partition(ii, 2) |> collect,
    )
end

function main(
    player_configurations = player_configurations;
    T_gt = 35,
    sim_kwargs = (;),
    interactive = false,
    visualize_time_steps = [1, 6, 11, 16, 21],
    savepath = "$project_root_dir/figures/highway/receding_horizon.pdf",
)
    setup = setup_highway(player_configurations; T = T_gt)
    @run_cached sim_result = RecedingHorizon.simulate(setup; sim_kwargs...)

    if !isnothing(visualize_time_steps)
        Makie.set_theme!()
        Makie.update_theme!(;
            Axis = (;
                topspinevisible = false,
                rightspinevisible = false,
                bottomspinevisible = false,
            ),
        )
        viz = visualize_receding_horizon(sim_result, visualize_time_steps)
        Makie.save(savepath, viz)
        viz
    end
end

player_configurations = [
    # Vehicle on the right lane wishing to merge left to go faster
    (;
        initial_lane = 1.0,
        initial_progress = -2,
        initial_speed = 0.2,
        target_speed = 0.4,
        speed_cost = 1.5,
        target_lane = -1.0,
        prox_cost = 0.3,
    ),
    # Fast vehicle from the back that would like to maintain its speed.
    (;
        initial_lane = -1.0,
        initial_progress = -3.0,
        initial_speed = 0.4,
        target_speed = 0.4,
        target_lane = -1.0,
        speed_cost = 1.5,
        prox_cost = 0.3,
    ),
    # Slow truck on the right lane
    (;
        initial_lane = 1.0,
        initial_progress = 2,
        initial_speed = 0.20,
        target_speed = 0.20,
        speed_cost = 1.0,
        target_lane = 1.0,
        prox_cost = 0.05,
    ),
    # Slow truck on the right lane
    (;
        initial_lane = 1.0,
        initial_progress = 4,
        initial_speed = 0.20,
        target_speed = 0.20,
        speed_cost = 1.0,
        target_lane = 1.0,
        prox_cost = 0.05,
    ),
    # Fast vehicle on the left lane wishing to merge back on the right lane and slow down
    (;
        initial_lane = -1.0,
        initial_progress = 6,
        initial_speed = 0.3,
        target_speed = 0.2,
        speed_cost = 1.5,
        target_lane = 1.0,
        prox_cost = 0.3,
    ),
]

end
