const project_root_dir = realpath(joinpath(@__DIR__, ".."))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "experiments/utils/MonteCarloStudy"))))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "test/utils"))))

using Distributed: Distributed

# TODO: Dependecies for online debugging. Factor out into another file
using Distances: Distances
using PartiallyObservedInverseGames.CostUtils: CostUtils
using PartiallyObservedInverseGames.InversePreSolve: InversePreSolve
using Statistics: Statistics

Distributed.@everywhere begin
    using Pkg: Pkg
    Pkg.activate($project_root_dir)
    union!(LOAD_PATH, $LOAD_PATH)

    using MonteCarloStudy: MonteCarloStudy
    using CollisionAvoidanceGame: CollisionAvoidanceGame
    using TestDynamics: TestDynamics
    using PartiallyObservedInverseGames.ForwardGame: IBRGameSolver, KKTGameSolver
    using PartiallyObservedInverseGames.InverseGames:
        InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game
end

import PartiallyObservedInverseGames.TrajectoryVisualization
using VegaLite: VegaLite
import LazyGroupBy: grouped

# Utils
include("utils/misc.jl")
include("utils/simple_caching.jl")
load_cache_if_not_defined!("unicycle_online")

#==================================== Forward Game Formulation =====================================#

T = 25

control_system =
    TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), TestDynamics.Unicycle(0.25)])

player_angles = let
    n_players = length(control_system.subsystems)
    map(eachindex(control_system.subsystems)) do ii
        angle_fraction = n_players == 2 ? pi / 2 : 2pi / n_players
        (ii - 1) * angle_fraction
    end
end

x0 = mapreduce(vcat, player_angles) do player_angle
    [unitvector(player_angle + pi); 0.1; player_angle + deg2rad(10)]
end

position_indices = mapreduce(vcat, eachindex(control_system.subsystems)) do subsystem_idx
    TestDynamics.state_indices(control_system, subsystem_idx)[1:2]
end

partial_state_indices = mapreduce(vcat, eachindex(control_system.subsystems)) do subsystem_idx
    TestDynamics.state_indices(control_system, subsystem_idx)[[1, 2, 4]]
end

player_cost_models_gt = map(enumerate(player_angles)) do (ii, player_angle)
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model(;
        player_idx = ii,
        control_system,
        T,
        goal_position = unitvector(player_angle),
        T_activate_goalcost = 1,
        fix_costs = (; state_goal = 0.1),
    )
end

#======================================== Monte Carlo Study ========================================#

## Dataset Generation
n_observation_sequences_per_noise_level = 5
# TODO: think about how to add different time windows to the dataset
# TODO: make sure that the window is aligned properly in evaluation:
#   - When we the window does not start at 1 then we would be comapring differnt time steps
observation_range = 1:(T ÷ 3)

@run_cached forward_solution_gt, dataset = MonteCarloStudy.generate_dataset(;#=@run_cached=#
    solve_args = (; solver = IBRGameSolver(), control_system, player_cost_models_gt, x0, T),
    # TODO restore exisiting noise levels
    noise_levels = unique([0:0.001:0.01; 0.01:0.005:0.03; 0.03:0.01:0.1]),
    #noise_levels = unique([0, 0.01]),
    n_observation_sequences_per_noise_level,
    observation_range,
)

## Estimation
estimator_setup = (;
    dataset,
    control_system,
    player_cost_models = player_cost_models_gt,
    solver_attributes = (; print_level = 1),
    T,
    cmin = 1e-3,
    player_weight_prior = nothing,
    pre_solve_kwargs = (; u_regularization = 1e-5),
    max_observation_error = nothing,
)
estimator_setup_partial =
    merge(estimator_setup, (; expected_observation = x -> x[partial_state_indices, :]))

# TODO: actually use the truncated setup ...
@run_cached estimates_conKKT =
    MonteCarloStudy.estimate(InverseKKTConstraintSolver(); estimator_setup...)
#@run_cached estimates_conKKT_partial =
#    MonteCarloStudy.estimate(InverseKKTConstraintSolver(); estimator_setup_partial...)
#@run_cached estimates_resKKT =
#    MonteCarloStudy.estimate(InverseKKTResidualSolver(); estimator_setup...)
#@run_cached estimates_resKKT_partial =
#    MonteCarloStudy.estimate(InverseKKTResidualSolver(); estimator_setup_partial...)

estimates = [
    estimates_conKKT;
    #    estimates_conKKT_partial
    #    estimates_resKKT
    #    estimates_resKKT_partial
]

demo_gt = merge((; player_cost_models_gt), forward_solution_gt)
errstats = map(estimates) do estimate
    MonteCarloStudy.estimator_statistics(
        estimate;
        dataset,
        demo_gt,
        position_indices,
        observation_range,
    )
end

frame = [-floor(1.5n_observation_sequences_per_noise_level), 0]
parameter_error_viz = errstats |> MonteCarloStudy.visualize_paramerr(; frame, round_x_axis = false)

# Debugging test
#=

d = dataset[end]
observation_horizon = T ÷ 3
y = d.x[:, (1:observation_horizon) .+ 5]
observation_model = (; d.σ, expected_observation = identity)

solver = InverseKKTResidualSolver()

# TODO: do this in a cleaner way. Would be nice to have the residual solver implement the exact same
# interface as the constraint solver by just doing pre-filtering etc inside the one-argument version
# of `solve_inverse_game`. Could probably just be a dispatch.
converged, sol = if solver isa InverseKKTConstraintSolver
    solve_inverse_game(
        solver,
        y;
        control_system,
        observation_model,
        player_cost_models = player_cost_models_gt,
        T,
        cmin = 1e-3,
        player_weight_prior = nothing, #[ones(4) / 4 for _ in 1:2],
        pre_solve_kwargs = (; u_regularization = 1e-5),
        max_observation_error = nothing,
    )
elseif solver isa InverseKKTResidualSolver
    smoothed_observation = let
        pre_solve_converged, pre_solve_solution = InversePreSolve.pre_solve(
            # pre-filter for baseline receives one extra state observation to avoid
            # unobservability of the velocity at the end-point.
            y,
            nothing;
            control_system,
            observation_model,
        )
        @assert pre_solve_converged
        # Filtered sequence is truncated to the original length to give all methods the same
        # number of data-points for inference.
        (; x = pre_solve_solution.x[:, 1:size(y, 2)], u = pre_solve_solution.u[:, 1:size(y, 2)])
    end
    converged, estimate = solve_inverse_game(
        solver,
        smoothed_observation.x,
        smoothed_observation.u;
        control_system,
        player_cost_models = player_cost_models_gt,
    )

    sol = merge(estimate, (; converged, smoothed_observation...))
    converged, sol
else
    error("Unknown solver.")
end
@assert converged

# visualization
let
    gt = TrajectoryVisualization.trajectory_data(control_system, dataset[begin].x)
    observation = TrajectoryVisualization.trajectory_data(control_system, y)
    estimate = TrajectoryVisualization.trajectory_data(control_system, sol.x)

    canvas = TrajectoryVisualization.visualize_trajectory(gt; group = "ground truth", legend = true)
    canvas =
        TrajectoryVisualization.visualize_trajectory(observation; canvas, group = "observation")
    canvas = TrajectoryVisualization.visualize_trajectory(estimate; canvas, group = "estimation")
    display(canvas)
end

function parameter_error(ps1, ps2)
    map(ps1, ps2) do p1, p2
        p1 = CostUtils.normalize(p1)
        p2 = CostUtils.normalize(p2)
        Distances.cosine_dist(p1, p2)
    end |> Statistics.mean
end

ws_gt = [m.weights for m in player_cost_models_gt]
@show parameter_error(ws_gt, sol.player_weights)
=#
