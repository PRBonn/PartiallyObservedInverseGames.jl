module CollisionAvoidanceGame

using JuMP: @NLconstraint, @NLexpression, @objective, @variable
using JuMPOptimalControl.CostUtils: symbol

export generate_player_cost_model

# TODO We should probably pass the `ProductSystem` and the player index here to compute the
# `state_indices`, `input_indices`, and `opponent_indices`.
function generate_player_cost_model(;
    T,
    state_indices,
    input_indices,
    goal_position,
    weights = (;
        state_proximity = 1,
        state_velocity = 1,
        control_Δv = 1,
        control_Δθ = 1,
    ),
    cost_prescaling = (;
        state_goal = 100, # The state_goal weight is assumed to be fixed.
        state_proximity = 0.1,
        state_velocity = 1,
        control_Δv = 10,
        control_Δθ = 1,
    ),
    prox_min_regularization = 0.1,
    T_activate_goalcost = T,
    # TODO: dirty hack
    opponent_indices = 1 in state_indices ? (5:8) : (1:4),
)
    function add_regularized_squared_distance!(opt_model, x_sub_ego, x_sub_opp)
        @NLexpression(
            opt_model,
            [t = 1:T],
            (x_sub_ego[1, t] - x_sub_opp[1, t])^2 +
            (x_sub_ego[2, t] - x_sub_opp[2, t])^2 +
            prox_min_regularization
        )
    end

    function add_objective!(opt_model, x, u; weights)
        T = size(x, 2)
        # TODO: Currently, this implementation is *not* agnostic to the order of players or
        # input and state dimensions of other players subsystems. Get these values from
        # somewhere else to make it agnostic:
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views x_sub_opp = x[opponent_indices, :]

        prox_cost = let
            d_sq = add_regularized_squared_distance!(opt_model, x_sub_ego, x_sub_opp)
            prox_cost = @variable(opt_model, prox_cost[t = 1:T])
            @NLconstraint(opt_model, [t = 1:T], prox_cost[t] == -log(d_sq[t]))
            prox_cost
        end

        J̃ = (;
            state_goal = sum(el -> el^2, x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position),
            state_proximity = sum(prox_cost),
            state_velocity = sum(el -> el^2, x_sub_ego[3, :]),
            control_Δv = sum(el -> el^2, u_sub_ego[1, :]),
            control_Δθ = sum(el -> el^2, u_sub_ego[2, :]),
        )
        @objective(
            opt_model,
            Min,
            sum(weights[k] * cost_prescaling[k] * J̃[k] for k in keys(weights)) +
            J̃.state_goal * cost_prescaling.state_goal * sum(weights) / length(weights)
        )
    end

    function add_objective_gradients!(opt_model, x, u; weights)
        n_states, T = size(x)
        n_controls = size(u, 1)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views x_sub_opp = x[opponent_indices, :]

        dprox_dxy = let
            d_sq = add_regularized_squared_distance!(opt_model, x_sub_ego, x_sub_opp)
            dproxdx = @variable(opt_model, [t = 1:T])
            @NLconstraint(
                opt_model,
                [t = 1:T],
                dproxdx[t] == -2 * (x_sub_ego[1, t] - x_sub_opp[1, t]) / d_sq[t]
            )
            dproxdy = @variable(opt_model, [t = 1:T])
            @NLconstraint(
                opt_model,
                [t = 1:T],
                dproxdy[t] == -2 * (x_sub_ego[2, t] - x_sub_opp[2, t]) / d_sq[t]
            )
            [dproxdx'; dproxdy']
        end

        # TODO: Technically this is missing the negative gradient on the opponents state but we
        # can't control that anyway (certainly not in OL Nash). Must be fixed for non-decoupled
        # systems and potentially FB Nash.
        dJdx = let
            dJ̃dx_sub = (;
                state_goal = [
                    zeros(2, T_activate_goalcost - 1) 2*(x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position)
                    zeros(2, T)
                ],
                state_proximity = [dprox_dxy; zeros(2, T)],
                state_velocity = [zeros(2, T); 2 * x_sub_ego[3, :]'; zeros(1, T)],
                control_Δv = zeros(size(x_sub_ego)),
                control_Δθ = zeros(size(x_sub_ego)),
            )
            dJdx_sub =
                sum(
                    weights[k] * cost_prescaling[symbol(k)] * dJ̃dx_sub[symbol(k)]
                    for k in keys(weights)
                ) +
                dJ̃dx_sub.state_goal * cost_prescaling.state_goal * sum(weights) / length(weights)
            [
                zeros(first(state_indices) - 1, T)
                dJdx_sub
                zeros(n_states - last(state_indices), T)
            ]
        end

        dJdu = let
            dJdu_sub =
                2 * [weights[:control_Δv], weights[:control_Δθ]] .*
                [cost_prescaling[:control_Δv], cost_prescaling[:control_Δθ]] .* u_sub_ego
            [
                zeros(first(input_indices) - 1, T)
                dJdu_sub
                zeros(n_controls - last(input_indices), T)
            ]
        end

        (; dx = dJdx, du = dJdu)
    end

    (; player_inputs = input_indices, weights, add_objective!, add_objective_gradients!)
end

end
