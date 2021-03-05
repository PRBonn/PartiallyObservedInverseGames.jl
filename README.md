# JuMPOptimalControl.jl

![build](https://github.com/lassepe/JuMPOptimalControl.jl/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/lassepe/JuMPOptimalControl.jl/branch/master/graph/badge.svg?token=FZoqGLI2gF)](https://codecov.io/gh/lassepe/JuMPOptimalControl.jl)

## Setup

You can install this package as you would with any other Julia package. Either
clone this repository manually or run the following code in [package mode]()
```julia
pkg> dev https://github.com/lassepe/JuMPOptimalControl.jl
```
which will install the package to `.julia/dev/JuMPOptimalControl`.

Beyond that, we use [DVC](https://dvc.org) for binary data version control.
This part of the setup is only required if you want to load our results as
binary data rather than reproducing them yourself be re-running the
experiments. DVC can be installed as follow:

1. Install [dvc](https://dvc.org/doc/install) with google drive support, e.g. `pip install dvc[gdrive]`
2. [Optional] Setup [git hooks](https://dvc.org/doc/command-reference/install#installed-git-hooks) to automate the process of checking out binary files: `dvc install`

Now you can download the binary data and figures by running `dvc pull`.

## Directory Layout

`TODO: Describe this step.`

## Reproducing Results

`TODO: Describe this step.`

## Citation

`TODO: Add paper with teaser.png`
