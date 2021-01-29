struct ProductSystem{T}
    "An iterable collection of systems."
    subsystems::T
end

function Base.getproperty(system::ProductSystem, sym::Symbol)
    if sym === :n_states
        sum(s -> s.n_states, system.subsystems)
    elseif sym === :n_controls
        sum(s -> s.n_controls, system.subsystems)
    else
        getfield(system, sym)
    end
end

# TODO: Implement DynamicsModelInterface.visualize_trajectory
function DynamicsModelInterface.visualize_trajectory(
    system::ProductSystem,
    x;
    canvas = Plots.plot(),
)
    for (subsystem_idx, subsystem) in enumerate(system.subsystems)
        @views x_sub = x[state_indices(system, subsystem_idx), :]
        DynamicsModelInterface.visualize_trajectory(subsystem, x_sub; canvas)
    end

    canvas
end

"Returns an iterable of state indices for the `subsystem_idx`th subsystem."
function state_indices(system::ProductSystem, subsystem_idx)
    # Note: starting from Julia 1.6 this can be done directly with a sum.
    idx_offset = mapreduce(i -> system.subsystems[i].n_states, +, 1:(subsystem_idx - 1); init = 0)
    idx_offset .+ (1:(system.subsystems[subsystem_idx].n_states))
end

"Returns an iterable of input indices for the `subsystem_idx`th subsystem."
function input_indices(system::ProductSystem, subsystem_idx)
    # Note: starting from Julia 1.6 this can be done directly with a sum.
    idx_offset = mapreduce(i -> system.subsystems[i].n_controls, +, 1:(subsystem_idx - 1); init = 0)
    idx_offset .+ (1:(system.subsystems[subsystem_idx].n_controls))
end

function DynamicsModelInterface.add_dynamics_constraints!(system::ProductSystem, opt_model, x, u)
    for (subsystem_idx, subsystem) in enumerate(system.subsystems)
        @views x_sub = x[state_indices(system, subsystem_idx), :]
        @views u_sub = u[input_indices(system, subsystem_idx), :]
        DynamicsModelInterface.add_dynamics_constraints!(subsystem, opt_model, x_sub, u_sub)
    end
end

function DynamicsModelInterface.add_dynamics_jacobians!(system::ProductSystem, opt_model, x, u)
    for (subsystem_idx, subsystem) in enumerate(system.subsystems)
        @views x_sub = x[state_indices(system, subsystem_idx), :]
        @views u_sub = u[input_indices(system, subsystem_idx), :]
        DynamicsModelInterface.add_dynamics_constraints!(subsystem, opt_model, x_sub, u_sub)
    end
end
