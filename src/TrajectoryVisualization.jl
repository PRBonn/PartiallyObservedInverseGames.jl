module TrajectoryVisualization
import VegaLite

export PlotsBackend, VegaLiteBackend, visualize_trajectory, visualize_trajectory_batch

#=========================== Visualization Interface for Dynamics  Models ==========================#

struct PlotsBackend end
struct VegaLiteBackend end

function visualize_trajectory end

"""
Visualizes a single trajectory `x` given as a matrix of states for a preset `control_system` and
visualziation `backend`.
"""
visualize_trajectory(control_system, x, backend;  canvas, kwargs...)

"""
Generic visualization of a single tarjectory `trajectory_data` via a given `backend`. Here, the
`trajectory_data` is provided in a tabular format and is self-contained. That is, the table-liek
format must have columns have positions `px` and `py`, time `t`, and player identifier `player`.
"""
visualize_trajectory(trajectory_data, backend; canvas, kwargs...)

"The multi-trajectory version of `visualize_trajectory`. Simply draws multiple trajectories on top
of eachother."
function visualize_trajectory_batch end

function trajectory_data end

"Converts the state matrix x into a tabular format of `trajectory_data` that can be handed to
visualize_trajectory"
trajectory_data(control_system, x, player)

#===================================== generic implementations =====================================#


function visualize_trajectory(control_system, x, backend::VegaLiteBackend; kwargs...)
    td = trajectory_data(control_system, x)
    visualize_trajectory(td, backend; kwargs...)
end

"The default-implementation of generic "
function visualize_trajectory(
    trajectory_data,
    ::VegaLiteBackend;
    canvas = VegaLite.@vlplot(),
    x_position_domain = extrema(s.px for s in trajectory_data) .+ (-0.01, 0.01),
    y_position_domain = extrema(s.py for s in trajectory_data) .+ (-0.01, 0.01),
    draw_line = true,
    legend = nothing,
)
    trajectory_visualizer =
        VegaLite.@vlplot(
            encoding = {
                x = {"px:q", scale = {domain = x_position_domain}, title = "Position x [m]"},
                y = {"py:q", scale = {domain = y_position_domain}, title = "Position y [m]"},
                order = "t:q",
                color = {"player:n", title = "Player", legend = legend},
            }
        ) +
        VegaLite.@vlplot(mark = {"point", shape = "circle", size = 25, clip = true, filled = true})

    if draw_line
        trajectory_visualizer += VegaLite.@vlplot(mark = {"line", clip = true})
    end

    canvas + (trajectory_data |> trajectory_visualizer)
end

function visualize_trajectory_batch(
    control_system,
    trajectory_batch,
    backend::VegaLiteBackend = VegaLiteBackend();
    canvas = VegaLite.@vlplot(opacity = {value = 0.2}, width = 200, height = 200),
    kwargs...,
)
    mapreduce(+, trajectory_batch; init = canvas) do x
        visualize_trajectory(control_system, x, backend; kwargs...)
    end
end

function visualize_trajectory_batch(
    trajectory_data_batch,
    backend::VegaLiteBackend = VegaLiteBackend();
    canvas = VegaLite.@vlplot(opacity = {value = 0.2}, width = 200, height = 200),
    kwargs...,
)
    mapreduce(+, trajectory_data_batch; init = canvas) do trajectory_data
        visualize_trajectory(trajectory_data, backend; kwargs...)
    end
end

end
