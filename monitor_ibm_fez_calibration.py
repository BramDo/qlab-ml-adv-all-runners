#!/usr/bin/env python3
"""Read-only ibm_fez calibration monitor for the q60 PBMC sentinel.

This script never submits circuits.  It queries IBM backend metadata once,
records a compact calibration snapshot, and invokes the existing Fire Opal
validate-only runner at most once for each newly observed calibration time.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qiskit_ibm_runtime import QiskitRuntimeService


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "fire_opal_pbmc68k_q60_modules_b4"
LATEST = ARTIFACT_DIR / "ibm_fez_calibration_latest.json"
HISTORY = ARTIFACT_DIR / "ibm_fez_calibration_history.jsonl"
ALERT = ARTIFACT_DIR / "ibm_fez_calibration_alert_latest.json"
LOCK = ARTIFACT_DIR / "ibm_fez_calibration_monitor.lock"
REFERENCE_VALIDATE = (
    ARTIFACT_DIR / "pbmc68k_q60_modules_b4_seed11_sentinel_provider_validate.json"
)
QCTRL_NOTEBOOK = Path(
    "/mnt/c/Users/Lenna/SynologyDrive/stackexchange/"
    "get-started-with-fire-opal-on-ibm-quantum.ipynb"
)
PYTHON = Path("/home/bram/.venvs/qiskit/bin/python")
BACKEND = "ibm_fez"
ACCOUNT = "default-ibm-cloud"
LOCK_STALE_SECONDS = 4 * 60 * 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def acquire_lock() -> int | None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        age = datetime.now(timezone.utc).timestamp() - LOCK.stat().st_mtime
        if age <= LOCK_STALE_SECONDS:
            return None
        LOCK.unlink()
    descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, f"pid={os.getpid()} captured_at_utc={utc_now()}\n".encode())
    return descriptor


def release_lock(descriptor: int) -> None:
    os.close(descriptor)
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return float(ordered[index])

    return {
        "count": len(ordered),
        "minimum": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "p90": percentile(0.9),
        "maximum": float(ordered[-1]),
    }


def calibration_snapshot() -> dict[str, Any]:
    service = QiskitRuntimeService(name=ACCOUNT)
    backend = service.backend(BACKEND)
    properties = backend.properties()
    x_errors: list[float] = []
    for gate in properties.gates:
        if gate.gate != "x":
            continue
        for parameter in gate.parameters:
            if parameter.name == "gate_error":
                x_errors.append(float(parameter.value))
    readout_errors: list[float] = []
    for qubit in range(backend.num_qubits):
        try:
            readout_errors.append(float(properties.readout_error(qubit)))
        except Exception:
            continue
    return {
        "schema_version": "1.0",
        "kind": "ibm_fez_calibration_snapshot",
        "captured_at_utc": utc_now(),
        "backend": BACKEND,
        "backend_num_qubits": int(backend.num_qubits),
        "backend_status": str(backend.status().status_msg),
        "calibration_last_update_utc": (
            properties.last_update_date.isoformat() if properties.last_update_date else None
        ),
        "x_gate_error": numeric_summary(x_errors),
        "readout_error": numeric_summary(readout_errors),
        "provider_calls": ["IBM backend metadata/properties/status"],
        "execution_attempted": False,
        "quantum_seconds_used": 0,
    }


def warning_occurrences(value: Any) -> list[str]:
    warnings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "warnings" and isinstance(child, list):
                warnings.extend(str(item) for item in child)
            warnings.extend(warning_occurrences(child))
    elif isinstance(value, list):
        for child in value:
            warnings.extend(warning_occurrences(child))
    return warnings


def collect_warnings(value: Any) -> list[str]:
    return sorted(set(warning_occurrences(value)))


def metric_change(
    current: dict[str, Any], previous: dict[str, Any]
) -> dict[str, Any]:
    ratios: dict[str, float] = {}
    for statistic in ("median", "p90"):
        now = current.get(statistic)
        before = previous.get(statistic)
        if (
            isinstance(now, (int, float))
            and isinstance(before, (int, float))
            and math.isfinite(float(now))
            and math.isfinite(float(before))
            and float(before) > 0
        ):
            ratios[statistic] = float(now) / float(before)
    return {
        "previous": previous,
        "current": current,
        "current_to_previous_ratio": ratios,
        "material_improvement_threshold": "median and p90 are each at least 20% lower",
        "materially_improved": bool(
            set(ratios) == {"median", "p90"}
            and all(value <= 0.8 for value in ratios.values())
        ),
    }


def warning_assessment(
    *, baseline_present: bool, current_present: bool, metrics: dict[str, Any]
) -> dict[str, Any]:
    if baseline_present and not current_present:
        outcome = "disappeared"
    elif current_present and bool(metrics.get("materially_improved")):
        outcome = "still_present_but_device_metrics_materially_improved"
    elif baseline_present and current_present:
        outcome = "still_present_not_materially_improved"
    elif not baseline_present and current_present:
        outcome = "appeared"
    else:
        outcome = "absent_in_baseline_and_current_validation"
    return {
        "baseline_warning_present": baseline_present,
        "current_warning_present": current_present,
        "outcome": outcome,
        "device_metric_change": metrics,
    }


def materially_lower(current: dict[str, Any], previous: dict[str, Any]) -> bool:
    comparisons: list[bool] = []
    for family in ("x_gate_error", "readout_error"):
        current_family = current.get(family, {})
        previous_family = previous.get(family, {})
        for statistic in ("median", "p90"):
            now = current_family.get(statistic)
            before = previous_family.get(statistic)
            if isinstance(now, (int, float)) and isinstance(before, (int, float)) and before > 0:
                comparisons.append(float(now) <= 0.8 * float(before))
    return bool(comparisons and all(comparisons))


def validate_new_calibration(snapshot: dict[str, Any], previous: dict[str, Any]) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = ARTIFACT_DIR / f"pbmc68k_q60_modules_b4_seed11_sentinel_{stamp}_qasm2.json.gz"
    report = ARTIFACT_DIR / f"pbmc68k_q60_modules_b4_seed11_sentinel_{stamp}_validate.json"
    command = [
        str(PYTHON),
        str(ROOT / "qiskit_qos_pbmc68k_q60_module_fireopal_validate.py"),
        "--phase",
        "sentinel",
        "--backend",
        BACKEND,
        "--validate",
        "--qiskit-account",
        ACCOUNT,
        "--qctrl-notebook",
        str(QCTRL_NOTEBOOK),
        "--bundle",
        str(bundle),
        "--output",
        str(report),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    validated = load_json(report) or {}
    unique_warnings = collect_warnings(validated)
    if validated:
        validated["warning_summary"] = {
            "deduplicated": True,
            "raw_occurrence_count": len(warning_occurrences(validated)),
            "unique_count": len(unique_warnings),
            "unique_warnings": unique_warnings,
        }
        atomic_json(report, validated)
    if completed.returncode != 0 or not report.exists():
        atomic_json(
            ALERT,
            {
                "kind": "ibm_fez_calibration_monitor_alert",
                "status": "validate_only_failed_no_retry_for_this_calibration",
                "captured_at_utc": utc_now(),
                "calibration_last_update_utc": snapshot.get("calibration_last_update_utc"),
                "error_type": "FireOpalValidateOnlyProcessError",
                "return_code": int(completed.returncode),
                "validate_report": str(report) if report.exists() else None,
                "execution_attempted": False,
                "quantum_seconds_used": 0,
            },
        )
        return
    reference = load_json(REFERENCE_VALIDATE) or {}
    baseline_warnings = collect_warnings(reference)
    x_warning = any("X gate error is much higher" in warning for warning in unique_warnings)
    measurement_warning = any(
        "measurement error is much higher" in warning for warning in unique_warnings
    )
    baseline_x_warning = any(
        "X gate error is much higher" in warning for warning in baseline_warnings
    )
    baseline_measurement_warning = any(
        "measurement error is much higher" in warning for warning in baseline_warnings
    )
    x_change = metric_change(
        snapshot.get("x_gate_error", {}), previous.get("x_gate_error", {})
    )
    measurement_change = metric_change(
        snapshot.get("readout_error", {}), previous.get("readout_error", {})
    )
    atomic_json(
        ALERT,
        {
            "kind": "ibm_fez_calibration_monitor_alert",
            "status": "new_calibration_validated",
            "captured_at_utc": utc_now(),
            "calibration_last_update_utc": snapshot.get("calibration_last_update_utc"),
            "validate_report": str(report),
            "validate_status": validated.get("status"),
            "unique_warnings": unique_warnings,
            "x_gate_warning_present": x_warning,
            "measurement_warning_present": measurement_warning,
            "warning_comparison": {
                "baseline_report_found": bool(reference),
                "x_gate": warning_assessment(
                    baseline_present=baseline_x_warning,
                    current_present=x_warning,
                    metrics=x_change,
                ),
                "measurement": warning_assessment(
                    baseline_present=baseline_measurement_warning,
                    current_present=measurement_warning,
                    metrics=measurement_change,
                ),
            },
            "device_metrics_materially_lower": materially_lower(snapshot, previous),
            "recommended_to_reconsider_sentinel": bool(
                validated.get("status") == "pass" and not x_warning and not measurement_warning
            ),
            "execution_attempted": False,
            "quantum_seconds_used": 0,
        },
    )


def main() -> int:
    descriptor = acquire_lock()
    if descriptor is None:
        return 0
    try:
        previous = load_json(LATEST) or {}
        try:
            snapshot = calibration_snapshot()
        except Exception as exc:
            atomic_json(
                ALERT,
                {
                    "kind": "ibm_fez_calibration_monitor_alert",
                    "status": "ibm_metadata_check_failed",
                    "captured_at_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "execution_attempted": False,
                    "quantum_seconds_used": 0,
                },
            )
            return 1
        changed = bool(
            previous.get("calibration_last_update_utc")
            and snapshot.get("calibration_last_update_utc")
            != previous.get("calibration_last_update_utc")
        )
        snapshot["calibration_changed_since_previous_check"] = changed
        append_jsonl(HISTORY, snapshot)
        atomic_json(LATEST, snapshot)
        if changed:
            validate_new_calibration(snapshot, previous)
        return 0
    finally:
        release_lock(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
