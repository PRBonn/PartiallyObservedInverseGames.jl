macro saveviz(expr::Expr)
    println(expr)
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
