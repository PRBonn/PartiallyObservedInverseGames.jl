module MonteCarloStudy

import Distributed
using Distributed: pmap

# TODO: think about at which level the @everywhere should happen. Probably outside of this.
import JuMPOptimalControl.InversePreSolve
using JuMPOptimalControl.ForwardGame: solve_game
using JuMPOptimalControl.InverseGames:
    InverseKKTConstraintSolver, InverseKKTResidualSolver, solve_inverse_game

import Random
import Statistics
import LinearAlgebra
import Distances
import VegaLite

import JuMPOptimalControl.CostUtils
import JuMPOptimalControl.DynamicsModelInterface
using JuMPOptimalControl.TrajectoryVisualization: visualize_trajectory, visualize_trajectory_batch
# TODO: ProgressMeter may need to be available for all workers
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
export visualize_bundle, visualize_paramerr, visualize_poserr
end
