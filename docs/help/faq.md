# Frequently asked questions

## General

**Does Quantum Mind need a quantum computer?**
No. Everything runs on an ordinary computer: the quantum-like and quantum-inspired models are
classical code, and the quantum models run on a built-in NumPy simulator. A quantum computer is only
needed if you choose to send circuits to IBM Quantum or Amazon Braket.

**Does "quantum-like" mean the brain is quantum?**
No. Quantum-like models borrow the mathematics of quantum probability because it describes certain
patterns in human answers (order effects, conjunction fallacies). They make no claim about physics in
the brain. See [concepts](../user_guide/concepts.md).

**Is there a quantum advantage?**
None is claimed. Every quantum and quantum-inspired model is compared with a classical baseline, and
the documentation reports when the classical method wins, which is often.

**Is any of this validated with real robots and real people?**
Not yet. The human data are published survey aggregates; all robotics results use simulated people.
See [validation status](validation.md) for what is and is not validated, and how to run a pilot.

## Installation

**Which Python versions are supported?**
Python 3.10 to 3.12, tested in CI.

**Which extras should I install?**

| Command | Adds |
|---|---|
| `pip install quantum-mind` | the core library (NumPy, SciPy) |
| `pip install "quantum-mind[viz]"` | Matplotlib and Plotly viewers, widgets |
| `pip install "quantum-mind[qiskit]"` | Qiskit, Aer and IBM Runtime for circuits |
| `pip install "quantum-mind[cloud]"` | IBM Runtime and Amazon Braket |
| `pip install "quantum-mind[rl]"` | Gymnasium environments |
| `pip install "quantum-mind[robotics]"` | Gymnasium and py_trees |
| `pip install "quantum-mind[all]"` | everything above |

**I get `ImportError: install quantum-mind[viz]`.**
The plotting back ends are optional. Install the extra named in the message.

**Qiskit fails to install on my machine.**
Qiskit publishes wheels for common platforms; on older systems upgrade `pip` first
(`python -m pip install --upgrade pip`). The library works without Qiskit; only circuit export and
hardware runs need it.

**Windows?**
The pure-Python parts work on Windows; the ROS 2 node follows ROS 2's own platform support.

## Using it

**How do I run on IBM Quantum hardware?**
Save your IBM Quantum API token with `qiskit_ibm_runtime.QiskitRuntimeService.save_account(...)`, then
call `quantum_mind.circuits.run(circuit, 'ibm:<backend name>', shots)`. See
[hardware](../user_guide/circuits.md). Hardware results are not included in the package.

**How large can the simulated circuits be?**
At most 22 qubits; up to about 10 qubits and a few thousand samples train in seconds to minutes.

**How do I fit my own data?**
Put your counts in a dictionary keyed by condition and call `quantum_mind.compare([...models...], data)`.
Tutorial 1 shows the format; each family's page gives its design.

**How do I use it on a robot?**
The decision functions are plain Python and middleware-free. `HumanModelService` wraps them with
JSON-friendly messages, and `integrations/ros2` provides a ROS 2 node. See tutorials 16 and 18.

**How do I cite it?**
Use the citation file (`CITATION.cff`) or the [citing page](../about/index.md).

## Documentation

**Are the numbers in the tutorials real?**
Yes. The tutorials are executed when the documentation is built; every printed number and figure is
the output of that run. The text states results in words, and an assertion in the tutorial checks each
stated claim, so the build fails if a claim stops being true.

**Why are some results worse for the quantum method?**
Because that is what the experiments show. Reporting it is deliberate.
