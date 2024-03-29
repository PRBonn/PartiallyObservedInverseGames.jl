module MonteCarloStudy

import Distributed
using Distributed: pmap

import PartiallyObservedInverseGames.InversePreSolve
using PartiallyObservedInverseGames.ForwardGame: solve_game
using PartiallyObservedInverseGames.InverseGames:
    InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game

import Random
import Statistics
import LinearAlgebra
import Distances
import VegaLite

import PartiallyObservedInverseGames.CostUtils
import PartiallyObservedInverseGames.DynamicsModelInterface
using PartiallyObservedInverseGames.TrajectoryVisualization: visualize_trajectory, visualize_trajectory_batch
using ProgressMeter: @showprogress
using VegaLite: @vlplot

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
