"""qlcog: quantum-like, quantum and quantum-inspired models, with classical baselines, Qiskit circuits
and a 3D Bloch-sphere viewer.

========================  ===========================================================================
Subpackage                Contents
========================  ===========================================================================
:mod:`qlcog.core`         linear algebra, the :class:`~qlcog.core.Model` base class, fitting,
                          comparison and model recovery
:mod:`qlcog.families`     quantum-like models of judgement and decision, eleven families with
                          classical baselines
:mod:`qlcog.applications` robotics: question domains, questioning designs, trust, intent resolution,
                          human-model ensemble
:mod:`qlcog.quantum`      quantum machine learning and algorithms on a batched simulator
:mod:`qlcog.inspired`     quantum-inspired algorithms that run on ordinary hardware
:mod:`qlcog.problems`     QUBO builders shared by the quantum and quantum-inspired solvers
:mod:`qlcog.circuits`     Qiskit circuits for the families and ``run()`` for simulators and hardware
:mod:`qlcog.viz`          Bloch-sphere viewer: trajectories, animations, interactive HTML, live updates
:mod:`qlcog.data`         published aggregate data sets
========================  ===========================================================================

The most used functions are re-exported at the top level: :func:`fit`, :func:`compare`,
:func:`recovery`, :class:`Model` and :class:`Param`.
"""
__version__ = '1.4.0'
from .core import fit, compare, recovery, Model, Param   # noqa: F401
