"""Published aggregate data used in the examples and tests.

These are the only human data shipped with the package; every value is taken from the source named
and was checked against secondary reports.

* :data:`CLINTON_GORE`: Gallup poll, 6-7 September 1997 (Moore, 2002; analysed by Wang and Busemeyer,
  2013): "Do you generally think [Bill Clinton / Al Gore] is honest and trustworthy?" Proportion yes:
  Clinton first 0.50, Gore first 0.68, Clinton second 0.57, Gore second 0.60.
"""
import numpy as np

#: Question-order effect in a Gallup poll (Clinton and Gore honesty questions).
CLINTON_GORE = dict(questions={'A': 'Clinton honest?', 'B': 'Gore honest?'},
                    rates={'A_first': 0.50, 'B_first': 0.68, 'A_second': 0.57, 'B_second': 0.60},
                    source='Moore (2002), Public Opinion Quarterly 66(1); Wang and Busemeyer (2013)')

def proportions_to_counts(props, n):
    """Turn proportions into binary counts.

    Used to fit multinomial models when only proportions and a sample size are known.

    Parameters
    ----------
    props : dict
        ``{condition: p}``.
    n : int
        Sample size per condition.

    Returns
    -------
    dict
        ``{condition: [round(n p), n - round(n p)]}``.

    Examples
    --------
    >>> from quantum_mind.data import proportions_to_counts
    >>> proportions_to_counts({'unknown': 0.36}, 98)['unknown'].tolist()
    [35, 63]
    """
    return {k: np.array([int(round(n * p)), n - int(round(n * p))]) for k, p in props.items()}
