# TODO: Maybe move this to a module.
# TODO: Currently this assumes the presence of a global variable `project_root_dir`, `result_cache`
# and `result_group`
import BSON
import Glob
import Dates

function clear_cache!()
    empty!(results_cache)
end

function save_cache!(result_group)
    save_path =
        joinpath(project_root_dir, "data/$result_group/results_cache-$(Dates.now(Dates.UTC)).bson")
    @assert !isfile(save_path)
    BSON.bson(save_path, results_cache)
end

function load_cache!(result_group)
    result_cache_file_list =
        Glob.glob("results_cache-*.bson", joinpath(project_root_dir, "data", result_group))
    if isempty(result_cache_file_list)
        false
    else
        global results_cache = BSON.load(last(result_cache_file_list))
        true
    end
end

macro run_cached(assigned_computation_expr)
    @assert assigned_computation_expr.head == :(=)
    var, fun = assigned_computation_expr.args

    quote
        $(esc(var)) =
            run_cached!(result_group, $(Meta.quot(var))) do
                $fun
            end
    end
end

function run_cached!(f, result_group, key; force_run = false)
    @assert !isempty(result_group)
    prefixed_key = Symbol(result_group, :., key)
    result = force_run ? f() : get(f, results_cache, prefixed_key)
    results_cache[prefixed_key] = result
    result
end

function load_cache_if_not_defined!(the_result_group)
    if !isdefined(Main, :results_cache)
        cache_file_found = load_cache!(the_result_group)
        if cache_file_found
            @info "Loaded cached results from file!"
        else
            @info "No persisted results cache file found. Resuming with an empty cache."
            global results_cache = Dict()
        end
    end
    global result_group = the_result_group
end
