module CostUtils

import JuMP
export symbol, normalize

symbol(s::Symbol) = s
symbol(s::JuMP.Containers.DenseAxisArrayKey) = only(s.I)

function normalize(weights)
    total = sum(weights)
    map(w -> w / total, weights)
end

function namedtuple(associative_array)
    nt_keys = Tuple(symbol.(keys(associative_array)))
    NamedTuple{nt_keys}(tuple(values(associative_array)...))
end

end
