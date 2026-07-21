from __future__ import annotations

import inspect
from argparse import Namespace

import pytest

import qiskit_qos_pbmc68k_q60_grid_d12_fireopal_pilot as pilot


def test_submit_requires_exact_confirmation_before_any_artifact_or_provider_call(
    tmp_path,
) -> None:
    args = Namespace(
        confirm_submit="yes",
        intent=tmp_path / "intent.json",
        receipt=tmp_path / "receipt.json",
    )

    with pytest.raises(pilot.PilotError, match="--confirm-submit"):
        pilot.submit_pilot(args)
    assert not args.intent.exists()
    assert not args.receipt.exists()


def test_mode_boundaries_are_separate_and_retrieval_cannot_execute() -> None:
    submit_source = inspect.getsource(pilot.submit_pilot)
    retrieve_source = inspect.getsource(pilot.retrieve_pilot)
    plan_source = inspect.getsource(pilot.plan_pilot)

    assert "fireopal.execute" in submit_source
    assert "fireopal.execute" not in retrieve_source
    assert "fireopal.execute" not in plan_source
    assert "fireopal.get_result" in retrieve_source
    assert "fireopal.get_result" not in submit_source
    assert "automatic_resubmission" in submit_source


def test_synthetic_probability_results_preserve_order_and_bounds() -> None:
    zero = "0" * pilot.PILOT_QUBITS
    raw = {"results": [{zero: 1.0} for _ in range(pilot.PILOT_CIRCUITS)]}
    manifest = []
    for circuit_index in range(pilot.PILOT_CIRCUITS):
        base_index = circuit_index // 3
        if base_index == 0:
            label = None
        else:
            label = -1.0 if (base_index - 1) % 2 == 0 else 1.0
        manifest.append(
            {
                "base_circuit_index": base_index,
                "measurement_basis": pilot.PILOT_MEASUREMENT_BASES[
                    circuit_index % 3
                ],
                "label": label,
            }
        )
    selected = [
        {
            "master_index": index,
            "measurement_basis": "Z",
            "pauli_mapping": [{"qubit": index, "pauli": "Z"}],
        }
        for index in range(24)
    ]
    result = pilot.validate_hardware_result(
        raw,
        manifest,
        {"seed_batch": {"selected_observables": selected}},
    )

    assert result["distribution_validation"]["passed"] is True
    assert result["distribution_validation"]["circuit_count"] == 195
    assert result["distribution_validation"]["semantics"] == ["probability"]
    assert result["observable_validation"]["expectation_count"] == 65 * 24
    assert result["observable_validation"]["minimum"] == 1.0
    assert result["observable_validation"]["maximum"] == 1.0


def test_cli_scope_matches_the_authorized_hardware_batch() -> None:
    parser = pilot.build_parser()
    submit = parser.parse_args(
        ["submit", "--confirm-submit", pilot.SUBMIT_CONFIRMATION]
    )

    assert submit.confirm_submit == pilot.SUBMIT_CONFIRMATION
    assert pilot.PILOT_BACKEND == "ibm_fez"
    assert pilot.PILOT_CIRCUITS == 195
    assert pilot.PILOT_SHOTS == 128
    assert pilot.PILOT_CIRCUITS * pilot.PILOT_SHOTS == 24_960
