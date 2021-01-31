module TestDynamics

import Plots
import JuMPOptimalControl.DynamicsModelInterface
import JuMP
using JuMP: @variable, @constraint, @NLconstraint

export visualize_trajectory, add_dynamics_constraints!, add_dynamics_jacobians!

include("unicycle.jl")
include("product_system.jl")

end
