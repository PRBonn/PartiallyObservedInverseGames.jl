module TestDynamics

import Plots
import VegaLite
import JuMPOptimalControl.DynamicsModelInterface
import JuMPOptimalControl.TrajectoryVisualization
import JuMP
using JuMP: @variable, @constraint, @NLconstraint

include("unicycle.jl")
include("product_system.jl")

end
