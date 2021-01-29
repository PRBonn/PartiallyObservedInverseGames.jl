module TestDynamics

import Plots
import JuMPOptimalControl.DynamicsModelInterface
using JuMP: @variable, @constraint, @NLconstraint

export visualize_trajectory, add_dynamics_constraints!, add_dynamics_jacobians!

include("unicycle.jl")

end
