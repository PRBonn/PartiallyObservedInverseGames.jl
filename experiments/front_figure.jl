# TODO: import/load all the dependencies

# visualize_highway
# forward_solution_gt
# dataset

"Returns a cost mapping that can be evaluated at (x, y) to get the running cost for that position."
function position_cost_mapping(;
    x_snapshot,
    player_idx,
    control_system,
    positional_cost_weights = (; state_proximity = 1, state_lane = 1),
    x_lane_center = nothing,
    y_lane_center = nothing,
    prox_min_regularization = 0.01,
)

    opponent_positions = let
        opponent_indices = filter(!=(player_idx), eachindex(control_system.subsystems))
        opponent_position_indices = map(opponent_indices) do jj
            TestDynamics.state_indices(control_system, jj)[1:2]
        end
        map(opponent_position_indices) do opp_position_idxs
            x_snapshot[opp_position_idxs]
        end
    end

    function (px, py)
        pos_ego = [px, py]

        J̃_position = (;
            state_lane = let
                # Note the switched indices. This is on purpose. The lane cost for the lane along y acts
                # on the x position and vice versa.
                x_lane_cost = isnothing(x_lane_center) ? 0 : (py - x_lane_center)^2
                y_lane_cost = isnothing(y_lane_center) ? 0 : (px - y_lane_center)^2
                x_lane_cost + y_lane_cost
            end,
            state_proximity = sum(opponent_positions) do pos_opponent
                prox_cost =
                    -log(LinearAlgebra.norm_sqr(pos_ego - pos_opponent) + prox_min_regularization)
            end,
        )
        sum(w * J̃_position[k] for (k, w) in pairs(positional_cost_weights))
    end

end

observation_idx = lastindex(dataset)
ground_truth = forward_solution_gt
demonstration = dataset[observation_idx]
prefiltered_observation = let
    d = estimates_resKKT[observation_idx]
    @assert d.observation_idx == observation_idx
    d.smoothed_observation
end

ground_truth_viz = visualize_highway(ground_truth.x)
raw_observations_viz = visualize_highway(demonstration.x; draw_line = false)
prefiltered_observation_viz = visualize_highway(prefiltered_observation.x; draw_line = true)

[ground_truth_viz raw_observations_viz prefiltered_observation_viz]

using VegaLite: @vlplot

cost_viz = let
    cost_map = position_cost_mapping(;
        x_snapshot = ground_truth.x[:, 1],
        player_idx = 2,
        control_system,
        y_lane_center = 0,
    )

    x_domain = -1:0.05:2
    y_domain = -4:0.05:14

    max_size = 500
    x_position_domain = x_domain
    y_position_domain = y_domain
    x_range = only(diff(extrema(x_position_domain) |> collect))
    y_range = only(diff(extrema(y_position_domain) |> collect))
    max_range = max(x_range, y_range)
    canvas = VegaLite.@vlplot(
        width = max_size * x_range / max_range,
        height = max_size * y_range / max_range
    )

    data = [(; x, y, cost = cost_map(x, y)) for x in x_domain, y in y_domain]
    data |>
    canvas + @vlplot(
        config = {view = {strokeWidth = 0, step = 13}},
        mark = {"point", shape = "square", filled = true},
        x = "x:q",
        y = "y:q",
        color = "cost:q"
    )
end
