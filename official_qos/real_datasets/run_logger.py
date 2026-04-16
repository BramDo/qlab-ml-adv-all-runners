from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload):
        ts = datetime.now(timezone.utc).isoformat()
        record = {"ts": ts, "event": event, **payload}
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
