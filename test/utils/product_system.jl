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

# Minor type piracy to patch https://github.com/JuliaLang/julia/pull/36222 in Julia 1.5.3. This
# won't be necessary starting from Julia 1.6.0-beta1.
if VERSION < v"1.6.0-"
    @inline function Base._cat_t(dims, T::Type{<:JuMP.GenericAffExpr}, X...)
        catdims = Base.dims2cat(dims)
        shape = Base.cat_shape(catdims, (), map(Base.cat_size, X)...)
        A = Base.cat_similar(X[1], T, shape)
        if count(!iszero, catdims) > 1
            fill!(A, zero(T))
        end
        return Base.__cat(A, shape, catdims, X...)
    end
end

function DynamicsModelInterface.add_dynamics_jacobians!(system::ProductSystem, opt_model, x, u)
    mapreduce(
        # diagonal pairwise concatenationo the elements in df and df_sub to comopse joint Jacobian
        # NOTE: dirty hack; scaling with 1 will make sure that d1 is promoted to an expression for
        # which Base.zero is defined.
        (df, df_sub) -> map((d1, d2) -> cat(1d1, d2; dims = (1, 2)), df, df_sub),
        enumerate(system.subsystems),
    ) do (subsystem_idx, subsystem)
        @views x_sub = x[state_indices(system, subsystem_idx), :]
        @views u_sub = u[input_indices(system, subsystem_idx), :]
        DynamicsModelInterface.add_dynamics_jacobians!(subsystem, opt_model, x_sub, u_sub)
    end
end

function DynamicsModelInterface.next_x(system::ProductSystem, x_t, u_t)
    mapreduce(vcat, enumerate(system.subsystems)) do (subsystem_idx, subsystem)
        @views x_t_sub = x_t[state_indices(system, subsystem_idx)]
        @views u_t_sub = u_t[input_indices(system, subsystem_idx)]
        DynamicsModelInterface.next_x(subsystem, x_t_sub, u_t_sub)
    end
end

#========================================== Visualization ==========================================#

function TrajectoryVisualization.visualize_trajectory(
    system::ProductSystem,
    x,
    backend::TrajectoryVisualization.PlotsBackend;
    canvas = Plots.plot(),
    kwargs...,
)
    for (subsystem_idx, subsystem) in enumerate(system.subsystems)
        @views x_sub = x[state_indices(system, subsystem_idx), :]
        TrajectoryVisualization.visualize_trajectory(subsystem, x_sub, backend; canvas, kwargs...)
    end

    canvas
end

function TrajectoryVisualization.visualize_trajectory(
    system::ProductSystem,
    x,
    backend::TrajectoryVisualization.VegaLiteBackend;
    canvas = VegaLite.@vlplot(),
    kwargs...,
)

    mapreduce(+, enumerate(system.subsystems); init = canvas) do (ii, subsystem)
        @views x_sub = x[state_indices(system, ii), :]
        TrajectoryVisualization.visualize_trajectory(
            subsystem,
            x_sub,
            backend;
            player = "P$ii",
            kwargs...,
        )
    end
end
