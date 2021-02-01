module CostUtils
import JuMP
export symbol
symbol(s::Symbol) = s
symbol(s::JuMP.Containers.DenseAxisArrayKey) = only(s.I)
end
