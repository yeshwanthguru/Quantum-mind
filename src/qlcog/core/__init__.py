"""Core layer: linear algebra of quantum-like models, the Model base class, fitting and comparison."""
from .linalg import *          # noqa: F401,F403
from .model import Param, Model
from .fit import FitResult, fit, compare, recovery, kl, tvd
