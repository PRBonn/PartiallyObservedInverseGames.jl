import FileIO
import JSON3
import CodecZlib
import PartiallyObservedInverseGames.DynamicsModelInterface

function unitvector(θ)
    [cos(θ), sin(θ)]
end

function rollout_strategy(control_system, x0, u)
    reduce(eachcol(u), init = x0[:, :]) do x_history, u_t
        x_t = x_history[:, end]
        x_tp = DynamicsModelInterface.next_x(control_system, x_t, u_t)
        [x_history x_tp]
    end
end

macro saveviz(expr::Expr)
    expr.head == :(=) || throw(ArgumentError("Expression must be an assigned computation."))
    var, fun = expr.args

    quote
        $(esc(expr))
        $(esc(var)) |>
        FileIO.save(joinpath(project_root_dir, "figures", result_group, "$($(Meta.quot(var))).pdf"))
    end
end

macro saveviz(sym::Symbol)
    quote
        $(esc(sym)) |>
        FileIO.save(joinpath(project_root_dir, "figures", result_group, "$($(Meta.quot(sym))).pdf"))
    end
end

macro save_json(expr)
    expr.head == :(=) || throw(ArgumentError("Expression must be an assigned computation."))
    var, fun = expr.args

    quote
        $(esc(expr))
        save_json($var, result_group, $(Meta.quot(var)))
    end
end



function save_json(result, result_group, key)
    result_file = joinpath(project_root_dir, "data/$result_group/$key.json.gz")
    open(io -> JSON3.write(io, result), CodecZlib.GzipCompressorStream, result_file, "w")
end
