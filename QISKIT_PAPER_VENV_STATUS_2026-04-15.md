## Qiskit in official paper venv

- Venv: `/home/bram/.venvs/qos-paper`
- Python: `3.12.3`
- Installed:
  - `qiskit==1.4.5`
  - `qiskit-aer==0.16.4`
- Verified retained paper stack:
  - `jax==0.8.1`
  - `jaxlib==0.8.1`

## Sanity simulation

- Backend: `aer_simulator`
- Circuit: `10`-qubit GHZ
- Statevector dimension: `1024`
- Shots: `4096`
- Counts:
  - `0000000000`: `2107`
  - `1111111111`: `1989`

## Repro commands

```bash
/home/bram/.venvs/qos-paper/bin/python -m pip install 'qiskit==1.4.5' 'qiskit-aer==0.16.4'
```

```bash
/home/bram/.venvs/qos-paper/bin/python - <<'PY'
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

q = 10
qc = QuantumCircuit(q, q)
qc.h(0)
for i in range(q - 1):
    qc.cx(i, i + 1)
qc.measure(range(q), range(q))

state = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
sim = AerSimulator()
result = sim.run(qc, shots=4096).result()
print(sim.name)
print(len(state.data))
print(result.get_counts())
PY
```
