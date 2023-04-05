viz_global_config = VegaLite.@vlfrag(legend = {orient = "top", padding = 0})
viz_defaults = (; width = 420, height = 300, frame = [-40, 0], scatter_opacity = 0.2)

color_map = Dict([
    "Ground Truth" => "black",
    "Observation" => "gray",
    "Ours Full" => "#4c78a8",
    "Ours Partial" => "#72b7b2",
    "Baseline Full" => "#e45756",
    "Baseline Partial" => "#f58518",
])

methods = ["Ours Full", "Ours Partial", "Baseline Full", "Baseline Partial"]
method_colors = [color_map[m] for m in methods]
method_color_encoding = VegaLite.@vlfrag(
    field = "estimator_name",
    type = "nominal",
    title = "Estimator",
    scale = {domain = methods, range = method_colors},
    legend = {symbolOpacity = 1.0},
)

function visualize_paramerr_over_noise(;
    scatter_opacity = viz_defaults.scatter_opacity,
    width = viz_defaults.width,
    height = viz_defaults.height,
    frame = viz_defaults.frame,
    y_label = "Mean Parameter Cosine Error",
    round_x_axis = true,
)
    @vlplot(
        config = viz_global_config,
        height = height,
        width = width,
        color = method_color_encoding,
        x = {
            "position_observation_error:q",
            title = "Postion Observation Error [m]",
            scale = {nice = round_x_axis},
        },
        transform = [
            {
                window = [
                    {field = "parameter_estimation_error", op = "median", as = "error_average"},
                    {field = "parameter_estimation_error", op = "q1", as = "error_band_lower"},
                    {field = "parameter_estimation_error", op = "q3", as = "error_band_upper"},
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
    @vlplot(mark = "line", y = "error_average:q",) +
    @vlplot(
        mark = {"errorband", extent = "iqr"},
        y = {"error_band_lower:q", title = y_label},
        y2 = "error_band_upper:q"
    )
end

function visualize_poserr_over_noise(;
    scatter_opacity = viz_defaults.scatter_opacity,
    width = viz_defaults.width,
    height = viz_defaults.height,
    frame = viz_defaults.frame,
    y_label = "Trajectory Reconstruction Error [m]",
    round_x_axis = true,
)
    @vlplot(
        config = viz_global_config,
        height = height,
        width = width,
        color = method_color_encoding,
        x = {
            "position_observation_error:q",
            title = "Postion Observation Error [m]",
            scale = {nice = round_x_axis},
        },
        transform = [
            {
                window = [
                    {field = "position_estimation_error", op = "median", as = "error_average"},
                    {field = "position_estimation_error", op = "q1", as = "error_band_lower"},
                    {field = "position_estimation_error", op = "q3", as = "error_band_upper"},
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
    @vlplot(mark = {"line"}, y = "error_average:q",) +
    @vlplot(
        mark = {"errorband", extent = "iqr"},
        y = {"error_band_lower:q", title = y_label},
        y2 = "error_band_upper:q",
    )
end

function visualize_paramerr_over_obshorizon(;
    width = 35,
    height = viz_defaults.height,
    y_label = "Mean Parameter Cosine Error",
)
    @vlplot(
        config = viz_global_config,
        width = width,
        height = height,
        mark = "boxplot",
        column = {
            "observation_horizon:o",
            title = "Observation Horizon [# time steps]",
            header =
                {titleOrient = "bottom", labelOrient = "bottom", labelPadding = height + 5},
            spacing = 2,
        },
        x = {"observation_model_type:n", title = nothing, axis = {orient = "top"}},
        y = {"parameter_estimation_error:q", title = y_label},
        color = method_color_encoding
    )
end

function visualize_poserr_over_obshorizon(;
    width = 35,
    height = viz_defaults.height,
    y_label = "Position Prediction Error [m]",
)
    @vlplot(
        width = width,
        height = height,
        config = viz_global_config,
        mark = "boxplot",
        column = {
            "observation_horizon:o",
            title = "Observation Horizon [# time steps]",
            header =
                {titleOrient = "bottom", labelOrient = "bottom", labelPadding = height + 5},
            spacing = 2,
        },
        x = {"observation_model_type:n", title = nothing, axis = {orient = "top"}},
        y = {"position_estimation_error:q", title = y_label},
        color = method_color_encoding
    )
end

function visualize_runtime_over_obshorizon(;
    width = 35,
    height = viz_defaults.height,
    y_label = "Runtime [s]",
)
    @vlplot(
        width = width,
        height = height,
        config = viz_global_config,
        mark = {type = "boxplot", clip = true},
        column = {
            "observation_horizon:o",
            title = "Observation Horizon [# time steps]",
            header =
                {titleOrient = "bottom", labelOrient = "bottom", labelPadding = height + 5},
            spacing = 2,
        },
        x = {"observation_model_type:n", title = nothing, axis = {orient = "top"}},
        y = {
            "runtime:q",
            title = y_label,
            scale = {type = "symlog", constant = 0.1, domain = [0, 0.75]},
        },
        color = method_color_encoding
    )
end
