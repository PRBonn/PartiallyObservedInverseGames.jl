# Note: This code assumes the presence of a global variable `project_root_dir`.
using BSON: BSON

function clear_cache!()
    empty!(results_cache)
end

function unload_cache!()
    global results_cache = nothing
end

function save_cache!(result_group; force = false)
    save_path = joinpath(project_root_dir, "data/$result_group/cache.bson")
    if !force
        !isfile(save_path) ||
            error("$save_path already exists. If you want to overwrite this file, call with kwarg \
                  `force = true`.")
    end
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
            $(esc(fun))
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
    global results_cache = if !isdefined(@__MODULE__, :results_cache) || isnothing(results_cache)
        loaded_cache = load_cache(the_result_group)
        if isnothing(loaded_cache)
            @info "No persisted results cache file found. Resuming with an empty cache."
            Dict()
        else
            @info "Loaded cached results for group \"$the_result_group\" from file!"
            loaded_cache
        end
    elseif isdefined(@__MODULE__, :results_cache) &&
           any(!startswith(string(k), "$the_result_group.") for k in keys(results_cache))
        error("Skipping. Cache contains results for another group. In order to proceed, call \
              `unload_cache!()`. If you want to keep the cached results, make sure to call \
              `save_cache!(result_group)` first.")
    elseif results_cache isa Dict
        @info "Cache for group \"$the_result_group\" present. No additional cache loaded."
        results_cache
    else
        error("Unknown cache type.")
    end

    global result_group = the_result_group
end
