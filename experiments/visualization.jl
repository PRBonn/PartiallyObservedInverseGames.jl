global_config = VegaLite.@vlfrag(legend = {orient = "top", padding = 0})
estimator_color_encoding = VegaLite.@vlfrag(
    field = "estimator_name",
    type = "nominal",
    title = "Estimator",
    scale = {
        domain = ["Ours Full", "Ours Partial", "Baseline Full", "Baseline Partial"],
        # Paired color scheme:
        # range = ["#a6cee3", "#1f78b4", "#bdbdbd", "#636363"],
        range = ["#4c78a8", "#72b7b2", "#e45756", "#f58518"],
    },
)

"Visualize all trajectory `estimates` along with the corresponding ground truth
`forward_solution_gt`"
function visualize_bundle(
    control_system,
    estimates,
    forward_solution_gt;
    filter_converged = false,
    kwargs...,
)
    position_domain = extrema(forward_solution_gt.x[1:2, :]) .+ (-0.01, 0.01)
    estimated_trajectory_batch = [e.x for e in estimates if !filter_converged || e.converged]
    visualize_trajectory_batch(
        control_system,
        estimated_trajectory_batch;
        position_domain,
        kwargs...,
    )
end

function visualize_paramerr(;
    scatter_opacity = 0.2,
    width = 500,
    height = 300,
    y_label = "Mean Parameter Cosine Error",
    frame = [-30, 0],
)
    @vlplot(
        config = global_config,
        height = height,
        width = width,
        color = estimator_color_encoding,
        x = {"position_observation_error:q", title = "Mean Absolute Postion Observation Error [m]"},
        transform = [
            {
                window = [
                    {field = "parameter_estimation_error", op = "mean", as = "error_average"},
                    {field = "parameter_estimation_error", op = "ci0", as = "error_band_lower"},
                    {field = "parameter_estimation_error", op = "ci1", as = "error_band_upper"},
                ],
                groupby = ["estimator_name"],
                frame = frame,
            },
        ]
    ) +
    @vlplot(
        mark = {"point", tooltip = {content = "data"}, opacity = scatter_opacity, filled = true},
        y = {"parameter_estimation_error:q", title = y_label},
    ) +
    @vlplot(
        mark = "line",
        #x = "σ:q",
        #y = {"parameter_estimation_error:q", aggregate = "median", title = y_label},
        y = "error_average:q",
    ) +
    @vlplot(
        mark = {"errorband", extent = "iqr"},
        #x = "σ:q",
        #y = {"parameter_estimation_error:q"},
        y = {"error_band_lower:q", title = y_label},
        y2 = "error_band_upper:q"
    )
end

function visualize_poserr(;
    scatter_opacity = 0.2,
    width = 500,
    height = 300,
    y_label = "Mean Absolute Position Prediciton Error [m]",
    frame = [-30, 0],
)
    @vlplot(
        config = global_config,
        height = height,
        width = width,
        color = estimator_color_encoding,
        x = {"position_observation_error:q", title = "Mean Absolute Postion Observation Error [m]"},
        transform = [
            {
                window = [
                    {field = "position_estimation_error", op = "mean", as = "error_average"},
                    {field = "position_estimation_error", op = "ci0", as = "error_band_lower"},
                    {field = "position_estimation_error", op = "ci1", as = "error_band_upper"},
                ],
                groupby = ["estimator_name"],
                frame = frame,
            },
        ]
    ) +
    @vlplot(
        mark = {"point", tooltip = {content = "data"}, opacity = scatter_opacity, filled = true},
        y = {
            "position_estimation_error:q",
            title = y_label,
            scale = {type = "symlog", constant = 0.01},
        },
        shape = {
            "converged:n",
            title = "Trajectory Reconstructable",
            legend = nothing,
            scale = {domain = [true, false], range = ["circle", "triangle-down"]},
        },
    ) +
    @vlplot(
        mark = "line",
        #x = "σ:q",
        #y = {"position_estimation_error:q", aggregate = "median", title = y_label}
        y = "error_average:q",
    ) +
    @vlplot(
        mark = {"errorband", extent = "iqr"},
        #x = "σ:q",
        #y = {"position_estimation_error:q", title = y_label},
        y = {"error_band_lower:q", title = y_label},
        y2 = "error_band_upper:q",
    )
end

