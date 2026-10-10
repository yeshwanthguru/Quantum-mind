# Quantum Perception–Dexterity Model (QPDM)

QPDM is a research prototype for **sensor-driven perception and dexterous robotics**. It provides one generic supervised model interface with two interchangeable feature backends:

- **Classical:** trigonometric sensor feature map plus a trainable softmax readout.
- **Qiskit:** parameterized sensor encoding and an entangling circuit, with local Pauli-Z expectations feeding the same kind of trainable softmax readout.

Applications define the labels and supply their own labeled sensor examples. The model does not assume a particular object, robot, grasp, or task.

## Use modes

Instantiate the same model separately for the capability you want:

- **Perception:** labels can represent scene states, object/contact classes, or motion categories.
- **Dexterity:** labels can represent grasp or manipulation primitives, such as pinch, power grasp, regrasp, or release.
- **Combined stack:** use a perception model’s outputs as additional inputs to a dexterity model. This composition is left to the application so each capability remains independently usable.

The current prototype predicts **categorical labels and their probabilities**. It does not process raw image pixels, generate continuous trajectories, output joint commands, or guarantee safe movement. A vision encoder and robot-specific controller can be connected around it later; low-level execution and safety remain separate.

## Sensor input

A model accepts one frame of named numeric sensor readings, or an ordered sequence of frames. Inputs may combine vision-derived features, depth, tactile/force readings, and robot state. For a sequence, the encoder uses the latest values, temporal means, and first-to-last changes. Missing known sensors are zero-filled. Sensor units should be consistent between training and prediction; supply characteristic input scales where appropriate.

## Install

The classical backend requires Python and NumPy. The Qiskit backend additionally requires Qiskit:

~~~bash
pip install numpy qiskit
~~~

Qiskit is imported only when the Qiskit backend is selected. Its current circuit path uses ideal local statevector simulation, not quantum hardware. Qiskit documents local statevector sampling and estimation through its [primitives API](https://docs.quantum.ibm.com/api/qiskit/primitives).

## Example: separate perception and dexterity models

~~~python
from qpdm import QuantumPerceptionDexterityModel

perception_examples = [
    {"depth_m": 0.42, "contact_force_n": 0.0},
    {"depth_m": 0.38, "contact_force_n": 0.1},
    {"depth_m": 0.12, "contact_force_n": 1.4},
    {"depth_m": 0.10, "contact_force_n": 1.8},
]
perception_labels = [
    "object_clear", "object_clear", "contact", "contact"
]
perception = QuantumPerceptionDexterityModel(
    labels=["object_clear", "contact"],
    mode="perception",
    backend="classical",
)
perception.fit(perception_examples, perception_labels)

dexterity_examples = [
    {"jaw_width_mm": 22, "grip_force_n": 2.0},
    {"jaw_width_mm": 19, "grip_force_n": 2.4},
    {"jaw_width_mm": 54, "grip_force_n": 8.0},
    {"jaw_width_mm": 50, "grip_force_n": 7.5},
]
dexterity_labels = ["pinch", "pinch", "power_grasp", "power_grasp"]
dexterity = QuantumPerceptionDexterityModel(
    labels=["pinch", "power_grasp"],
    mode="dexterity",
    backend="qiskit",
    n_qubits=4,
)
dexterity.fit(dexterity_examples, dexterity_labels)

scene = perception.predict_proba({"depth_m": 0.11, "contact_force_n": 1.6})
skill = dexterity.predict({"jaw_width_mm": 20, "grip_force_n": 2.2})
print(scene)
print(skill.label, skill.probabilities)
~~~

Each training example is either a mapping such as {"depth_m": 0.31, "normal_force_n": 1.4} or an ordered list of such mappings. Training examples and targets must have equal length. Configure one model per output vocabulary; the mode indicates the intended capability and does not hard-code its label set.

## Classical versus Qiskit comparison

Train both backends with the **same train/validation split, sensor preprocessing, labels, and metrics**. Compare task accuracy/F1, probability calibration, robustness to missing/noisy sensors, inference latency, and model size. Use multiple random seeds. The Qiskit version currently runs on a classical statevector simulator, so it tests a circuit-based representation; it does not establish quantum advantage. Hardware execution would add sampling noise, latency, and device constraints.

The classical feature map and Qiskit circuit are intentionally distinct representations with a shared supervised readout interface. A rigorous comparison should report that distinction and include capacity-matched conventional baselines.

## Research status and limitations

QPDM is an initial, generic scaffold for developing the perception–dexterity direction. The quantum-inspired/quantum-circuit hypothesis is that circuit feature interactions may help when multimodal sensor evidence is ambiguous or context-dependent. That remains to be tested on suitable robotics datasets against strong classical baselines. No performance or novelty claim is made by the code alone.

The Qiskit backend uses a small, fixed feature-encoding circuit; the learned parameters are in the classical readout. It is therefore a **hybrid Qiskit model**, not a fully variational quantum neural network. Current output is discrete; continuous dexterity control and safety validation are future stack components.
