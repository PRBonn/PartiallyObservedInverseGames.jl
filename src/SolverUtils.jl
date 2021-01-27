module SolverUtils

import JuMP

export get_values, get_model_values, set_solver_attributes!, init_if_hasproperty!

get_values(; jump_vars...) = (; map(((k, v),) -> k => JuMP.value.(v), collect(jump_vars))...)
get_model_values(model, symbols...) = get_values(; map(sym -> sym => model[sym], symbols)...)

function set_solver_attributes!(model, ; silent, solver_attributes...)
    silent ? JuMP.set_silent(model) : JuMP.unset_silent(model)
    foreach(
        ((k, v),) -> JuMP.set_optimizer_attribute(model, string(k), v),
        pairs(solver_attributes),
    )
end

function init_if_hasproperty!(v, init, sym)
    if hasproperty(init, sym)
        JuMP.set_start_value.(v, getproperty(init, sym))
    end
end

end
