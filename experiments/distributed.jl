"Launch workers locally and remotely. Note that you may have to load the code on the
launched workers explicity with `Distributed.@everywhere ...`."
function start_workers(;
    n_remote = :auto,
    n_local = 0,
    # NOTE: only the ssh tunnel is relevant for connecting to IPB remotely. `topology =
    # master_worker` seems neccessary to get ssh multi-plexing to work.
    config = (; tunnel = true, topology = :master_worker),
)
    usable(n) = n == :auto || n > 0

    if usable(n_remote)
        # rechenknecht node only works in HULKs network (VPN)
        Distributed.addprocs([
            # ("bonn-201-remote", n_remote),
            # ("bonn-202-remote", n_remote),
            # ("bonn-203-remote", n_remote),
            # ("bonn-204-remote", n_remote),
            # ("bonn-221-remote", n_remote),
            # ("bonn-222-remote", n_remote),
            # ("bonn-223-remote", n_remote),
            # ("bonn-224-remote", n_remote),
            ("bonn-student-81-remote", n_remote),
            ("bonn-student-82-remote", n_remote),
            ("bonn-student-83-remote", n_remote),
            ("bonn-student-84-remote", n_remote),
            ("bonn-student-85-remote", n_remote),
            ("bonn-student-86-remote", n_remote),
            ("bonn-student-87-remote", n_remote),
            #("bonn-student-89-remote", n_remote),
        ]; config...)
    end

    if usable(n_local)
        Distributed.addprocs(n_local)
    end

    Distributed.workers()
end

"Stop all workers."
function stop_workers()
    Distributed.rmprocs(Distributed.workers())
end
