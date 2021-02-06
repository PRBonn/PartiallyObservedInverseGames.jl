module TestUtils

import Random
import JuMP
using Test: @test

export test_inverse_solution, test_inverse_model, add_noise

function noisy_observation(observation_model, x; rng = Random.MersenneTwister(1))
    ŷ = observation_model.expected_observation(x)
    ŷ + randn(rng, size(ŷ)) .* observation_model.σ
end

function test_inverse_solution(weights_est, weights_true; atol = 1e-2, verbose = false)
    w_total_est = sum(weights_est)
    w_total_true = sum(weights_true)

    for k in keys(weights_true)
        verbose && @info k
        @test isapprox(weights_est[k] / w_total_est, weights_true[k] / w_total_true; atol = atol)
    end
end

# TODO: this should be named differently since the JuMP.termination_status test is now trivially
# performed via `converged`.
function test_inverse_model(inverse_model, observation_model, x̂, y)
    @test JuMP.termination_status(inverse_model) in (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL)
    ŷ = observation_model.expected_observation(x̂)
    ê_sq = sum((ŷ .- y) .^ 2)
    @test JuMP.objective_value(inverse_model) <= ê_sq + 1e-2
end

end
