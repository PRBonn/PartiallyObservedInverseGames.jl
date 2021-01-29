using Test: @test, @testset

using JuMP: @objective
using JuMPOptimalControl.ForwardGame: IBRGameSolver, solve_game

unique!(push!(LOAD_PATH, joinpath(@__DIR__, "utils")))
import TestUtils
import TestDynamics

control_system = TestDynamics.ProductSystem([TestDynamics.Unicycle(), TestDynamics.Unicycle()])

@testset "Product Dynamics" begin
    @test control_system.n_states == 8
    @test control_system.n_controls == 4

    @test all(i in TestDynamics.state_indices(control_system, 1) for i in 1:4)
    @test all(i in TestDynamics.state_indices(control_system, 2) for i in 5:8)
    @test all(i in TestDynamics.input_indices(control_system, 1) for i in 1:2)
    @test all(i in TestDynamics.input_indices(control_system, 2) for i in 3:4)
end

# TODO: avoid name collisions of weight keys in the inverse game setting.
function generate_player_cost_model(;
    state_indices,
    input_indices,
    goal_position,
    weights = (; state_velocity = 10, state_goal = 0.1, control_Δv = 100, control_Δθ = 10),
)

    function objective(x, u; weights)
        # TODO: Currently, this implementation is *not* agnostic to the order of players or
        # input and state dimensions of other players subsystems. Get these values from
        # somewhere else to make it agnostic:
        @views x_sub = x[state_indices, :]
        @views u_sub = u[input_indices, :]

        weights[:state_velocity] * sum(el -> el^2, x_sub[3, :]) +
        weights[:state_goal] * sum(el -> el^2, x_sub[1:2, :] .- goal_position) +
        weights[:control_Δv] * sum(el -> el^2, u_sub[1, :]) +
        weights[:control_Δθ] * sum(el -> el^2, u_sub[2, :])
    end

    function objective_gradients(x, u; weights)
        # TODO: This typic probably does not allow to assign expressioins later. Make a non-inplace
        # version or widen the type of dJdx, dJdu.
        dJdx = zeros(size(x))
        dJdu = zeros(size(u))
        @views x_sub = x[state_indices, :]
        @views u_sub = u[input_indices, :]
        @views dJdx_sub = dJdx[state_indices, :]
        @views dJdu_sub = dJdu[input_indices, :]
        dJdx_sub[3, :] .+= 2 * weights[:state_velocity] * x_sub[3, :]
        dJdx_sub[1:2, :] .+= 2 * weights[:state_goal] * (x_sub[1:2, :] .- goal_position)
        dJdu_sub[1, :] .+= 2 * weights[:control_Δv] * u_sub[1, :]
        dJdu_sub[2, :] .+= 2 * weights[:control_Δθ] * u_sub[2, :]

        (; dx = dJdx, du = dJdu)
    end

    (;
        player_inputs = input_indices,
        weights,
        objective,
        objective_gradients,
        add_objective! = (opt_model, args...; kwargs...) ->
            @objective(opt_model, Min, objective(args...; kwargs...)),
        add_objective_gradients! = (opt_model, args...; kwargs...) ->
            objective_gradients(args...; kwargs...),
    )
end

player_cost_models = let
    cost_model_p1 = generate_player_cost_model(;
        state_indices = 1:4,
        input_indices = 1:2,
        goal_position = [1, 0],
    )
    cost_model_p2 = generate_player_cost_model(;
        state_indices = 5:8,
        input_indices = 3:4,
        goal_position = [0, 1],
    )

    (cost_model_p1, cost_model_p2)
end

x0 = vcat([-1, 0, 0, 0], [0, -1, 0, pi / 4])
T = 100

ibr_converged, ibr_solution, ibr_models =
    solve_game(IBRGameSolver(), control_system, player_cost_models, x0, T)
