# TODO: Maybe move this to a module.
import BSON
import Glob
import Dates

function clear_cache!()
    empty!(results_cache)
end

function save_cache!()
    save_path = joinpath(@__DIR__, "../data/results_cache-$(Dates.now(Dates.UTC)).bson")
    @assert !isfile(save_path)
    BSON.bson(save_path, results_cache)
end

function load_cache!()
    result_cache_file_list = Glob.glob("results_cache-*.bson", joinpath(@__DIR__, "../data/"))
    if isempty(result_cache_file_list)
        false
    else
        global results_cache = BSON.load(last(result_cache_file_list))
        true
    end
end

macro run_cached(assigned_computation_expr)
    @assert assigned_computation_expr.head == :(=)
    var_name, fun_call = assigned_computation_expr.args

    quote
        run_cached!($(Meta.quot(var_name))) do
            $fun_call
        end
    end
end

function run_cached!(f, key; force_run = false)
    result = force_run ? f() : get(f, results_cache, key)
    results_cache[key] = result
    result
end

if !isdefined(Main, :results_cache)
    cache_file_found = load_cache!()
    if cache_file_found
        @info "Loaded cached results from file!"
    else
        @info "No persisted results cache file found. Resuming with an emtpy cache."
        global results_cache = Dict()
    end
end

