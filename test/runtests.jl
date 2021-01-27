using Test: @testset

@testset "Tests" begin
    include("lq_control.jl")
    include("nonlq_control.jl")
    include("ol_nash_game.jl")
end
