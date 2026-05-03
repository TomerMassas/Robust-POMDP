"""
JSON export of a solver run as a chronological event list.

The Python UI (viz/app.py) replays these events to show the algorithm's
execution step by step.

Schema:
    {
      "metadata": { "timestamp": ..., "n_events": ..., ...extra },
      "events":   [ { "type": "sim_start", ... }, ... ]
    }
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from robust_pomdp.debug.logger import SolverLog, num_events


def export_log_to_json(log: SolverLog,
                       path: str | Path,
                       *,
                       metadata: dict | None = None
                       ) -> Path:
    """Serialize a SolverLog to a JSON file.

    Args:
        log: SolverLog whose events will be written.
        path: output file path.
        metadata: extra metadata (problem name, params, etc.) merged into
            the base metadata (timestamp, n_events).

    Returns:
        The resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    base_meta: dict = {
        "timestamp": datetime.now().isoformat(),
        "n_events":  num_events(log),
    }
    full_meta = {**base_meta, **(metadata or {})}

    events_json = [
        {"type": ev.type, **ev.data}
        for ev in log.events
    ]

    out = {
        "metadata": full_meta,
        "events":   events_json,
    }

    with path.open("w") as f:
        json.dump(out, f, indent=2)

    return path
