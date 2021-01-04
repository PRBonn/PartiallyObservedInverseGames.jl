get_model_values(model, symbols...) = (; map(sym -> sym => JuMP.value.(model[sym]), symbols)...)

"The performance index for the forward optimal control problem."
function forward_objective(x, u; Q, R)
    T = last(size(x))
    sum(x[:, t]' * Q * x[:, t] + u[:, t]' * R * u[:, t] for t in 1:T)
end

"The performance index for the inverse optimal control problem."
function inverse_objective(x; x̂)
    T = last(size(x))
    sum(sum((x̂[:, t] - x[:, t]) .^ 2) for t in 1:T)
end
