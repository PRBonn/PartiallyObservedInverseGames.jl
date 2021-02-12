module JuMPOptimalControl
include("DynamicsModelInterface.jl")
include("TrajectoryVisualization.jl")
include("JuMPUtils.jl")
include("CostUtils.jl")
include("ForwardOptimalControl.jl")
include("InverseOptimalControl.jl")
include("ForwardGame.jl")
include("InverseGames.jl")
end
