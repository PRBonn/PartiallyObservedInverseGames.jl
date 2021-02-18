module TrajectoryVisualization
import VegaLite

export PlotsBackend, VegaLiteBackend, visualize_trajectory, visualize_trajectory_batch

#=========================== Visualization Interface for Dynamics  Models ==========================#

struct PlotsBackend end
struct VegaLiteBackend end
function visualize_trajectory end

# Setting default visualization backend:
visualize_trajectory(system, x; kwargs...) =
    visualize_trajectory(system, x, PlotsBackend(); kwargs...)

#================================= Trajectory Batch Visualization ==================================#

function visualize_trajectory_batch(
    control_system,
    trajectory_batch,
    backend::VegaLiteBackend = VegaLiteBackend();
    canvas = VegaLite.@vlplot(opacity = {value = 0.05}, width = 300, height = 300),
    kwargs...
)
    mapreduce(+, trajectory_batch; init = canvas) do x
        visualize_trajectory(control_system, x, backend; kwargs...)
    end
end

end
