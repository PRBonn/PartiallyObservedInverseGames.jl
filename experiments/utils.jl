function unitvector(θ)
    [cos(θ), sin(θ)]
end

function rollout_strategy(control_system, x0, u)
    reduce(eachcol(u), init = x0[:,:]) do x_history, u_t
        x_t = x_history[:, end]
        x_tp = DynamicsModelInterface.next_x(control_system, x_t, u_t)
        [x_history x_tp]
    end
end

macro saveviz(expr::Expr)
    @assert expr.head == :(=)
    var, fun = expr.args

    quote
        $(esc(expr))
        $(esc(var)) |> VegaLite.save(joinpath(project_root_dir, "figures/$($(Meta.quot(var))).pdf"))
    end
end

macro saveviz(sym::Symbol)
    quote
        $(esc(sym)) |> VegaLite.save(joinpath(project_root_dir, "figures/$($(Meta.quot(sym))).pdf"))
    end
end
