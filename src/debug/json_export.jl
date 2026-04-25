"""
JSON export of a solver run as a chronological event list.

The Python UI (`viz/app.py`) replays these events to show the algorithm's
execution step by step.

Schema:
    {
      "metadata": { problem, params, state/action/obs names, ... },
      "events":   [ { "type": "sim_start", ... }, ... ]
    }
"""

using JSON
using Dates

"""
    export_log_to_json(log, path; metadata=Dict())

Serialize an event-based SolverLog to a JSON file.

Args:
    log: SolverLog whose events will be written
    path: output file path
    metadata: extra metadata (problem name, params, etc.)
"""
function export_log_to_json(log::SolverLog, path::String; metadata::Dict=Dict())
    base_meta = Dict(
        "timestamp"  => string(now()),
        "n_events"   => num_events(log),
    )
    full_meta = merge(base_meta, metadata)

    # Serialize events: convert the Symbol type to a String
    events_json = [
        merge(Dict("type" => String(ev.type)), ev.data)
        for ev in log.events
    ]

    out = Dict(
        "metadata" => full_meta,
        "events"   => events_json,
    )

    open(path, "w") do io
        JSON.print(io, out, 2)
    end

    return path
end
