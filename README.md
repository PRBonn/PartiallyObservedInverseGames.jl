# JuMPOptimalControl.jl

![build](https://github.com/lassepe/JuMPOptimalControl.jl/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/lassepe/JuMPOptimalControl.jl/branch/master/graph/badge.svg?token=FZoqGLI2gF)](https://codecov.io/gh/lassepe/JuMPOptimalControl.jl)

## Setup

You can install this package as you would with any other Julia package. Either
clone this repository manually or run the following code in [package
mode](https://docs.julialang.org/en/v1/stdlib/Pkg/)
```julia
pkg> dev https://github.com/lassepe/JuMPOptimalControl.jl
```
which will install the package to `.julia/dev/JuMPOptimalControl`.

Beyond that, we use [DVC](https://dvc.org) for binary data version control.
This part of the setup is only required if you want to load our results as
binary data rather than reproducing them yourself be re-running the
experiments. DVC can be installed as follow:

1. Install [dvc](https://dvc.org/doc/install) with google drive support, e.g.
   `pip install dvc[gdrive]`
2. [Optional] Setup [git
   hooks](https://dvc.org/doc/command-reference/install#installed-git-hooks) to
   automate the process of checking out binary files: `dvc install`

Now you can download the binary data and figures by running `dvc pull`.

## Directory Layout

- `src/` contains the implementations of our method and the baseline for
  inverse planning. Beyond that it contains implementations of forward game
  solvers and visualization utilities. The most important files here are

- `test/` contains unit and integration tests for the code in `src/`

- `experiments/` contains the code for reproducing the Monte Carlo study for
  the running example (`experiments/unicycle.jl`) and the highway overtaking
  scenario (`experiments/highway.jl`).

- After setting up `dvc` as described above and running `dvc pull` the
  directory `data/` contains the binary data (as `.bson` file) of our results
  as well as their visualization (as `.pdf` file).

## Reproducing Results

The results of the Monte Carlo study can be reproduced by running the
corresponding scripts in `experiments/`:

- 2-Player running example of collision avoidance: `experiments/unicycle.jl`
- 5-Player highway overtaking scenario: `experiments/highway.jl`

### Caching

Both scripts will check for cached results in `data/`. If cached results have
been found, they will be loaded and the figures will be reproduced from this
data. In order to reproduce results from scratch you will have to clear the
cache first by calling `clear_cache!()` (implemented in
`experiments/utils/simple_caching.jl`). Alternatively, you can remove the
`@run_cached`  macro in front the function calls in the experiment to disable
caching for that call.

### Distributed Experiments

Running a large scale Monte Carlo study can take a substantial amount of time.
Thus, this package uses
[`Distributed.jl`](https://docs.julialang.org/en/v1/stdlib/Distributed/) for
parallelization. If there are multiple workers registered in the worker pool,
the experiment scripts will automatically parallelize since all
heavy lifting is implemented using
[`Distributed.pmap`](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.pmap).
Workers can run on the same machine, on a remote cluster, or even both. The
only requirement is that all code can be loaded on the remote worker. This can
be achieved by mounting the repository to a shared directory that is available
from all nodes in the (potentially heterogeneous) cluster or by utilizing
`rsync`. A suit of useful utility functions for this task can also be found in
[Distributor.jl](https://github.com/lassepe/Distributor.jl).

## Citation

`TODO: Add paper with teaser.png`
