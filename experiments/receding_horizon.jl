module RecedingHorizon
const project_root_dir = realpath(joinpath(@__DIR__, ".."))

#using PartiallyObservedInverseGames.ForwardGame: ForwardGame
using Random: Random
using ProgressMeter: ProgressMeter
using GLMakie: GLMakie
using Makie: Makie

include("utils/preamble.jl")
include("utils/unicycle_online_setup.jl")

function visualize_receding_horizon(sim_result, player_markers = ['●', '■'], domain = 1.5)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; aspect = 1, limits = ((-domain, domain), (-domain, domain)))

    time_slider = Makie.Slider(fig[2, 1]; range = eachindex(sim_result.estimator_steps))
    step = Makie.@lift sim_result.estimator_steps[$(time_slider.value)]

    for (ii, p) in pairs(sim_result.position_indices)
        marker = player_markers[ii]
        buffer_size = Makie.@lift size($step.y_full, 2)

        ground_truth = [Makie.Point2f(x[p]) for x in eachcol(sim_result.forward_solution_gt.x)]
        prediction = Makie.@lift [Makie.Point2f(x[p]) for x in eachcol($step.x)]
        observations_full =
            Makie.@lift [Makie.Point2f(y_full[p]) for y_full in eachcol($step.y_full)]

        # line and point for ground truth
        Makie.lines!(ax, ground_truth; color = "green")
        Makie.scatter!(ax, Makie.@lift ground_truth[$step.t]; color = "green", marker)

        # line and point for prediction
        Makie.lines!(ax, prediction; color = "blue")
        Makie.scatter!(ax, Makie.@lift $prediction[$buffer_size]; color = "blue", marker)

        # raw observations in the buffer
        Makie.scatter!(ax, observations_full; color = ("gray", 0.5), marker)
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

    observations_full = forward_solution_gt.x .+ σ * randn(rng, size(forward_solution_gt.x))
    observations_partial = observation_model.expected_observation(observations_full)

    # TODO warm-start with last solution (also costs weights)
    estimator_steps = NamedTuple[]
    ProgressMeter.@showprogress for t in t_start:(T_gt - T_predict)
        obs_window = max((t - T_obs + 1), 1):t
        @assert length(obs_window) <= T_obs
        y = observations_partial[:, obs_window]
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

        push!(estimator_steps, (; t, y_full = observations_full[:, obs_window], solution...))
    end

    (;
        estimator_steps,
        forward_solution_gt,
        position_indices = position_indices |> ii -> Iterators.partition(ii, 2) |> collect,
    )
end
end
