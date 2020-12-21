# TODO: Maybe write a simple converter for ControlSystems.jl

# Optimization
import JuMP: JuMP, @constraint, @objective, @variable
import Ipopt
using LinearAlgebra: I

# Visualization
import ElectronDisplay, Plots; Plots.plotly();

#= Describe a simple forward LQR problem using a collocation style method  =#
model = JuMP.Model(Ipopt.Optimizer)
n_stages = 100
n_states = 2
n_controls = 1
A = [1 1; 0 1]
B = collect([0 1]')
Q = I
R = 100I
x0 = [10., 10.]
# Decision variables
@variable(model, x[1:n_states, 1:n_stages])
@variable(model, u[1:n_controls, 1:n_stages])
# Initial condition
@constraint(model, initial_condition, x[:, 1] .== x0)
# Dynamics are setup via constraints
@constraint(model, dynamics[t=1:n_stages-1], x[:, t+1] .== A * x[:, t] + B * u[:, t])
# A simple quadratic objective
JuMP.@objective(model, Min, sum(x[:, t]' * Q * x[:, t] + u[:, t]' * R * u[:, t] for  t in 1:n_stages))
JuMP.optimize!(model)

# Visualization
plt = Plots.plot(JuMP.value.(x)[1, :])
