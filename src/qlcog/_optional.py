"""Clear error messages for optional dependencies."""
import importlib

_EXTRA = {'qiskit': 'qiskit', 'qiskit_aer': 'qiskit', 'qiskit_ibm_runtime': 'qiskit', 'braket': 'cloud',
          'matplotlib': 'viz', 'plotly': 'viz', 'ipywidgets': 'viz', 'anywidget': 'viz', 'seaborn': 'viz'}


def require(module, extra=None):
    """Import `module` or raise ImportError naming the pip extra that provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        extra = extra or _EXTRA.get(module.split('.')[0], 'all')
        raise ImportError('%s is needed for this feature: pip install "qlcog[%s] @ '
                          'git+https://github.com/yeshwanthguru/quantum-cognition-robotics"' % (module, extra)) from e
