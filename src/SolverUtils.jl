module SolverUtils

import JuMP

get_model_values(model, symbols...) = (; map(sym -> sym => JuMP.value.(model[sym]), symbols)...)

function set_solver_attributes!(model, ; silent, solver_attributes...)
    silent ? JuMP.set_silent(model) : JuMP.unset_silent(model)
    foreach(
        ((k, v),) -> JuMP.set_optimizer_attribute(model, string(k), v),
        pairs(solver_attributes),
    )
end

end
