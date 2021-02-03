module JuMPUtils

import JuMP

export get_values, get_model_values, set_solver_attributes!, init_if_hasproperty!

get_values(; jump_vars...) = (; map(((k, v),) -> k => JuMP.value.(v), collect(jump_vars))...)
get_model_values(opt_model, symbols...) =
    get_values(; map(sym -> sym => opt_model[sym], symbols)...)

function set_solver_attributes!(opt_model; solver_attributes...)
    foreach(
        ((k, v),) -> JuMP.set_optimizer_attribute(opt_model, string(k), v),
        pairs(solver_attributes),
    )
end

function init_if_hasproperty!(v, init, sym; default = nothing)
    init_value = hasproperty(init, sym) ? getproperty(init, sym) : default
    if !isnothing(init_value)
        JuMP.set_start_value.(v, init_value)
    end
end

end
