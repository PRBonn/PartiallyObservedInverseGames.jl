module TestDynamics

import Plots
import VegaLite
import PartiallyObservedInverseGames.DynamicsModelInterface
import PartiallyObservedInverseGames.TrajectoryVisualization
import JuMP
using JuMP: @variable, @constraint, @NLconstraint

include("unicycle.jl")
include("product_system.jl")

end
