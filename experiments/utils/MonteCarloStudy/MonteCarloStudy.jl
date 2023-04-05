module MonteCarloStudy

using Distributed: Distributed
using Distributed: pmap

import PartiallyObservedInverseGames.InversePreSolve
using PartiallyObservedInverseGames.ForwardGame: solve_game
using PartiallyObservedInverseGames.InverseGames:
    InverseKKTConstraintSolver,
    InverseKKTResidualSolver,
    AugmentedInverseKKTResidualSolver,
    solve_inverse_game

using Random: Random
using Statistics: Statistics
using LinearAlgebra: LinearAlgebra
using Distances: Distances
using VegaLite: VegaLite

import PartiallyObservedInverseGames.CostUtils
import PartiallyObservedInverseGames.DynamicsModelInterface
using PartiallyObservedInverseGames.TrajectoryVisualization:
    visualize_trajectory, visualize_trajectory_batch
using ProgressMeter: @showprogress
using VegaLite: @vlplot
using Setfield: @set
using JuMP: JuMP

#======================================== Generate Dataset =========================================#

include("dataset_generation.jl")
export generate_dataset

include("estimation.jl")
export estimate

include("forward_augmentation.jl")
export augment_with_forward_solution

include("estimator_statistics.jl")
export estimator_statistics

include("visualization.jl")
export visualize_paramerr, visualize_poserr
end
