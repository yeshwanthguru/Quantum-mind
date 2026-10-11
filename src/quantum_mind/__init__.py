"""quantum_mind: quantum robotics: computing, sensing and cognition, each against its classical baseline.

* :mod:`quantum_mind.computing`: thread 1, quantum computing for robot algorithms: robot problems as QUBOs, classical
  baselines and a Qiskit QAOA adapter.
* :mod:`quantum_mind.sensing`: thread 2, quantum sensing for robots: quantum limits, accelerometer models and Kalman fusion.
* Thread 3, quantum-probability cognition: :mod:`quantum_mind.belief`, :mod:`quantum_mind.families` and
  :mod:`quantum_mind.applications` below.

* :mod:`quantum_mind.core`: linear algebra for states and density matrices, the :class:`~quantum_mind.core.Model`
  base class, fitting, comparison and model recovery.
* :mod:`quantum_mind.families`: quantum-like models of question order, inference, trust and perception, four
  families with classical baselines.
* :mod:`quantum_mind.belief`: COBRA, a density-matrix belief filter whose measurement basis is rotated by the
  previous sensor frame, with an online mode, the Bayes filter as a classical reference and a minimum-cost
  action rule.
* :mod:`quantum_mind.applications`: the robot decision layer: question domains, intent resolution, the
  human-model ensemble, ask or act, calibration and conformal sets, orchestration gates, question
  planning, personalisation, trust-aware hand-over and allocation, and multimodal fusion.
* :mod:`quantum_mind.envs`: Gymnasium environments with simulated people.
* :mod:`quantum_mind.circuits`: Qiskit circuits for the families and ``run()`` for simulators and hardware.
* :mod:`quantum_mind.quantum`: the state-vector simulator behind the viewer and the circuit exports.
* :mod:`quantum_mind.problems`: the QUBO container used by trust-aware task allocation.
* :mod:`quantum_mind.viz`: Bloch-sphere viewer: trajectories, animations, interactive HTML, live updates.
* :mod:`quantum_mind.data`: published aggregate data sets.

The most used functions are re-exported at the top level: :func:`fit`, :func:`compare`,
:func:`recovery`, :class:`Model` and :class:`Param`.
"""
__version__ = '3.0.0'
from .core import fit, compare, recovery, Model, Param, bootstrap, OnlinePersonModel   # noqa: F401
