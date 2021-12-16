# TODO: it would be much cleaner if this was wrapped in a function but I'm not sure how caching
# would work in that case (at least not if the function was living in module).

## Dataset Generation
n_observation_sequences_per_instance = 40

@run_cached dataset = MonteCarloStudy.generate_dataset_noise_sweep(;
    solve_args = (; solver = IBRGameSolver(), control_system, player_cost_models_gt, x0, T),
    noise_levels = unique([0:0.001:0.01; 0.01:0.005:0.03; 0.03:0.01:0.1]),
    n_observation_sequences_per_instance,
)

## Estimation
estimator_setup = (;
    dataset,
    control_system,
    player_cost_models = player_cost_models_gt,
    solver_attributes = (; print_level = 1),
)
estimator_setup_partial =
    merge(estimator_setup, (; expected_observation = x -> x[partial_state_indices, :]))
@run_cached estimates_conKKT =
    MonteCarloStudy.estimate(InverseKKTConstraintSolver(); estimator_setup...)
@run_cached estimates_conKKT_partial =
    MonteCarloStudy.estimate(InverseKKTConstraintSolver(); estimator_setup_partial...)
@run_cached estimates_resKKT =
    MonteCarloStudy.estimate(InverseKKTResidualSolver(); estimator_setup...)
@run_cached estimates_resKKT_partial =
    MonteCarloStudy.estimate(InverseKKTResidualSolver(); estimator_setup_partial...)

## Forward Solution Augmentation
augmentor_kwargs = (;
    solver = KKTGameSolver(),
    control_system,
    player_cost_models_gt,
    x0,
    T,
    solver_attributes = (; print_level = 1),
)
@run_cached augmented_estimates_resKKT =
    MonteCarloStudy.augment_with_forward_solution(estimates_resKKT; augmentor_kwargs...)
@run_cached augmented_estimates_resKKT_partial =
    MonteCarloStudy.augment_with_forward_solution(estimates_resKKT_partial; augmentor_kwargs...)

@save_json estimates = [
    estimates_conKKT
    estimates_conKKT_partial
    augmented_estimates_resKKT
    augmented_estimates_resKKT_partial
]

## Error Ststistics Computation
@save_json errstats = map(estimates) do estimate
    MonteCarloStudy.estimator_statistics(
        estimate;
        player_cost_models_gt,
        position_indices,
        window_type = :observation,
        T_predict = 0,
    )
end

frame = [-floor(1.5n_observation_sequences_per_instance), 0]
@saveviz parameter_error_viz =
    errstats |> MonteCarloStudy.visualize_paramerr_over_noise(; frame, round_x_axis = false)
@saveviz position_error_viz =
    errstats |> MonteCarloStudy.visualize_poserr_over_noise(; frame, round_x_axis = false)
