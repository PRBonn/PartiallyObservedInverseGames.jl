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
    canvas = VegaLite.@vlplot(opacity = {value = 0.025}, width = 800, height = 800),
)
    mapreduce(+, trajectory_batch; init = canvas) do x
        visualize_trajectory(control_system, x, backend; is_observation = true)
    end
end

end
