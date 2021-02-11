module DynamicsModelInterface

export add_dynamics_constraints!,
    add_dynamics_jacobians!, visualize_trajectory, PlotsBackend, VegaLiteBackend

function add_dynamics_constraints! end
function add_dynamics_jacobians! end
function visualize_trajectory end
struct PlotsBackend end
struct VegaLiteBackend end
DynamicsModelInterface.visualize_trajectory(system, x; kwargs...) =
    DynamicsModelInterface.visualize_trajectory(system, x, PlotsBackend(); kwargs...)

end
