"""quantum_handover: trust-aware robot hand-over with Quantum Mind's quantum-like trust model.

* :mod:`quantum_handover.policies`: robot policies (quantum-like, Markov, two fixed baselines) and
  populations of simulated people.
* :mod:`quantum_handover.benchmark`: every policy against both populations.
* :mod:`quantum_handover.sim`: PyBullet simulation of a KUKA arm handing a cube to a person, saved as a
  GIF (needs the ``sim`` extra).
* :mod:`quantum_handover.ros2_node`: a ROS 2 node around the same policy (needs a ROS 2 installation).
"""
from .policies import POLICIES, make_policy, quantum_like_people, markov_people, person_state
from .benchmark import run_benchmark, to_markdown

__all__ = ['POLICIES', 'make_policy', 'quantum_like_people', 'markov_people', 'person_state',
           'run_benchmark', 'to_markdown']
__version__ = '0.1.0'
