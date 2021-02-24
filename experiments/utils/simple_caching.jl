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

function load_cache(result_group)
    result_cache_file_list =
        Glob.glob("results_cache-*.bson", joinpath(project_root_dir, "data", result_group))
    if isempty(result_cache_file_list)
        nothing
    else
        BSON.load(last(result_cache_file_list))
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

function load_cache_if_not_defined!(the_result_group; filly_emtpy = true)
    global results_cache =
        if isdefined(Main, :results_cache) &&
           any(!startswith(string(k), "$the_result_group.") for k in keys(results_cache))
            error("Skipping. Cache contains results for another group. Safe and/or clear first.")
        elseif !isdefined(Main, :results_cache) || filly_emtpy && isempty(results_cache)
            loaded_cache = load_cache(the_result_group)
            if isnothing(loaded_cache)
                @info "No persisted results cache file found. Resuming with an empty cache."
                Dict()
            else
                @info "Loaded cached results for group \"$the_result_group\" from file!"
                loaded_cache
            end
        else
            @info "Results for group \"$the_result_group\" present. No additional cache loaded."
            results_cache
        end

    global result_group = the_result_group
end
