using Query: @groupby, @map, @join, key
using DataFrames: DataFrame

highway_worstcases =
    errstats |>
    @groupby(_.estimator_name) |>
    @map({
        estimator_name = key(_),
        worst_case_observation_idx =
            getindex(_, argmax(_.position_estimation_error)).observation_idx,
    }) |>
    @join(
        estimates,
        Any[_.worst_case_observation_idx, _.estimator_name],
        Any[_.observation_idx, _.estimator_name],
        {_.estimator_name, _.worst_case_observation_idx, __.x}
    ) |>
    DataFrame
