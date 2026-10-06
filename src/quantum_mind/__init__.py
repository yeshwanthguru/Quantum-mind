"""quantum_mind: quantum-like, quantum and quantum-inspired models, with classical baselines, Qiskit circuits
and a 3D Bloch-sphere viewer.

========================  ===========================================================================
Subpackage                Contents
========================  ===========================================================================
:mod:`quantum_mind.core`         linear algebra, the :class:`~quantum_mind.core.Model` base class, fitting,
                          comparison and model recovery
:mod:`quantum_mind.families`     quantum-like models of judgement and decision, eleven families with
                          classical baselines
:mod:`quantum_mind.applications` robotics: question domains, questioning designs, trust, intent resolution,
                          human-model ensemble
:mod:`quantum_mind.quantum`      quantum machine learning and algorithms on a batched simulator
:mod:`quantum_mind.inspired`     quantum-inspired algorithms that run on ordinary hardware
:mod:`quantum_mind.problems`     QUBO builders shared by the quantum and quantum-inspired solvers
:mod:`quantum_mind.circuits`     Qiskit circuits for the families and ``run()`` for simulators and hardware
:mod:`quantum_mind.viz`          Bloch-sphere viewer: trajectories, animations, interactive HTML, live updates
:mod:`quantum_mind.data`         published aggregate data sets
========================  ===========================================================================

The most used functions are re-exported at the top level: :func:`fit`, :func:`compare`,
:func:`recovery`, :class:`Model` and :class:`Param`.
"""
__version__ = '2.0.1'
from .core import fit, compare, recovery, Model, Param   # noqa: F401
