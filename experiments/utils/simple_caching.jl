# TODO: Maybe move this to a module.
# TODO: Currently this assumes the presence of a global variable `project_root_dir`, `result_cache`
# and `result_group`
import BSON

function clear_cache!()
    empty!(results_cache)
end

function unload_cache!()
    global results_cache = nothing
end

function save_cache!(result_group)
    save_path = joinpath(project_root_dir, "data/$result_group/cache.bson")
    @assert !isfile(save_path)
    BSON.bson(save_path, results_cache)
end

function load_cache(result_group)
    result_cache_file = joinpath(project_root_dir, "data/$result_group/cache.bson")
    if !isfile(result_cache_file)
        nothing
    else
        BSON.load(result_cache_file)
    end
end

macro run_cached(assigned_computation_expr)
    @assert assigned_computation_expr.head == :(=)
    var, fun = assigned_computation_expr.args

    quote
        $(esc(var)) = run_cached!(result_group, $(Meta.quot(var))) do
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
    global results_cache = if !isdefined(Main, :results_cache) || isnothing(results_cache)
        loaded_cache = load_cache(the_result_group)
        if isnothing(loaded_cache)
            @info "No persisted results cache file found. Resuming with an empty cache."
            Dict()
        else
            @info "Loaded cached results for group \"$the_result_group\" from file!"
            loaded_cache
        end
    elseif isdefined(Main, :results_cache) &&
           any(!startswith(string(k), "$the_result_group.") for k in keys(results_cache))
        error("Skipping. Cache contains results for another group. Save and/or unload first.")
    elseif results_cache isa Dict
        @info "Cache for group \"$the_result_group\" present. No additional cache loaded."
        results_cache
    else
        error("Unknown cache type.")
    end

    global result_group = the_result_group
end
