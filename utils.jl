get_model_values(model, symbols...) = (; map(sym -> sym => JuMP.value.(model[sym]), symbols)...)
