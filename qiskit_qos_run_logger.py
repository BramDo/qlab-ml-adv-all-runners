from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _run_log_path() -> Path | None:
    value = os.environ.get("QISKIT_QOS_RUN_LOG")
    if value is None or not value.strip():
        return None
    return Path(value.strip())


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return repr(value)


def log_run_event(event: str, **payload: Any) -> None:
    path = _run_log_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "event": event,
        **{key: _json_ready(value) for key, value in payload.items()},
    }
    line = json.dumps(record, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
