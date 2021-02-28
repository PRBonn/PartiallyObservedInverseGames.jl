# TODO: import/load all the dependencies

# visualize_highway
# forward_solution_gt
# dataset
import LinearAlgebra
import ElectronDisplay
import FileIO
using VegaLite: @vlplot

"Returns a cost mapping that can be evaluated at (x, y) to get the running cost for that position."
function position_cost_mapping(;
    x_sequence,
    player_idx,
    control_system,
    positional_cost_weights,
    x_lane_center = nothing,
    y_lane_center = nothing,
    prox_min_regularization = 0.01,
    fix_cost_weights = (; state_lane = 0.1),
)

    opponent_positions = let
        opponent_indices = filter(!=(player_idx), eachindex(control_system.subsystems))
        opponent_position_indices = map(opponent_indices) do jj
            TestDynamics.state_indices(control_system, jj)[1:2]
        end
        map(opponent_position_indices) do opp_position_idxs
            x_sequence[opp_position_idxs, :]
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
            state_proximity = sum(opponent_positions) do pos_opponent_sequence
                sum(eachcol(pos_opponent_sequence)) do pos_opponent
                    -log(LinearAlgebra.norm_sqr(pos_ego - pos_opponent) + prox_min_regularization)
                end
            end,
        )
        sum(w * J̃_position[k] for (k, w) in pairs(positional_cost_weights)) +
        sum(w * J̃_position[k] for (k, w) in pairs(fix_cost_weights))
    end

end

"Generates a visualiztion of the inferred positional stage for a player."
function cost_viz(
    player_idx,
    player_config = player_configurations[player_idx],
    player_weights = estimates_conKKT_partial[observation_idx].player_weights[player_idx],
    ;
    x_sequence = ground_truth.x[:, 1:1],
    control_system = control_system,
    show_y_label = false,
)
    cost_map = position_cost_mapping(;
        x_sequence,
        player_idx,
        control_system,
        y_lane_center = player_config.target_lane,
        positional_cost_weights = (; state_proximity = player_config.prox_cost),
    )

    x_domain = -1:0.09:2
    y_domain = -4:0.09:14

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

    normalized_cost = let
        data = [(; x, y, cost = cost_map(x, y)) for x in x_domain, y in y_domain]
        cmin, cmax = extrema(d.cost for d in data)
        [(; d.x, d.y, cost = (d.cost - cmin) / (cmax - cmin)) for d in data]
    end

    normalized_cost |>
    canvas + @vlplot(
        mark = {"point", shape = "square", filled = true, strokeWidth = 0},
        x = {"x:q", title = "Position x [m]"},
        y = {"y:q", title = "Position y [m]", axis = show_y_label ? true : nothing},
        color = {
            "cost:q",
            scale = {scheme = "viridis"},
            legend = {gradientLength = 200},
            title = "Normalized Cost",
        },
    )
end

observation_idx = lastindex(dataset)
ground_truth = forward_solution_gt
demonstration = dataset[observation_idx]
prefiltered_observation = let
    d = estimates_resKKT[observation_idx]
    @assert d.observation_idx == observation_idx
    d.smoothed_observation
end

# @saveviz highway_frontfig_gt_viz = visualize_highway(ground_truth.x)
# 
# @saveviz highway_frontfig_observation_viz = visualize_highway(demonstration.x; draw_line = false)
# @saveviz prefiltered_observation_viz =
#     visualize_highway(prefiltered_observation.x; draw_line = true)
# 
# @saveviz highway_frontfig_player_cost_viz =
#     @vlplot(resolve = {scale = {color = "shared"}, axis = {position_y = "shared"}}) +
#     hcat(cost_viz(1; show_y_label = true), cost_viz(2), cost_viz(3), cost_viz(4), cost_viz(5))
# @saveviz highway_frontfig_estimate_viz =
#     visualize_highway(estimates_conKKT_partial[observation_idx].x)
# 
# @saveviz highway_frontfig_cost1 = cost_viz(1; show_y_label = true)
# @saveviz highway_frontfig_cost5 = cost_viz(5; show_y_label = true)
