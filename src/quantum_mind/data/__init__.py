"""Published aggregate data used in the examples and tests.

These are the only human data shipped with the package; every value is taken from the source named
and was checked against secondary reports.

* :data:`CLINTON_GORE`: Gallup poll, 6-7 September 1997 (Moore, 2002; analysed by Wang and Busemeyer,
  2013): "Do you generally think [Bill Clinton / Al Gore] is honest and trustworthy?" Proportion yes:
  Clinton first 0.50, Gore first 0.68, Clinton second 0.57, Gore second 0.60.
* :data:`PRISONERS_DILEMMA`: Shafir and Tversky (1992): proportion defecting when the opponent is known
  to have defected 0.97, known to have cooperated 0.84, unknown 0.63.
* :data:`TWO_STAGE_GAMBLE`: Tversky and Shafir (1992), 98 students: proportion choosing to play the
  second gamble after winning the first 0.69, after losing 0.59, outcome unknown 0.36.
* :data:`LINDA`: Tversky and Kahneman (1983), succinct version: 85% of participants rated "bank teller
  and active in the feminist movement" as more probable than "bank teller".
"""
import numpy as np

#: Question-order effect in a Gallup poll (Clinton and Gore honesty questions).
CLINTON_GORE = dict(questions={'A': 'Clinton honest?', 'B': 'Gore honest?'},
                    rates={'A_first': 0.50, 'B_first': 0.68, 'A_second': 0.57, 'B_second': 0.60},
                    source='Moore (2002), Public Opinion Quarterly 66(1); Wang and Busemeyer (2013)')
#: Disjunction effect in the Prisoner's Dilemma.
PRISONERS_DILEMMA = dict(conditions={'known_1': 0.97, 'known_2': 0.84, 'unknown': 0.63},
                         labels={'known_1': 'opponent defected', 'known_2': 'opponent cooperated', 'act': 'defect'},
                         source='Shafir and Tversky (1992), Cognitive Psychology 24(4)')
#: Disjunction effect in the two-stage gamble.
TWO_STAGE_GAMBLE = dict(conditions={'known_1': 0.69, 'known_2': 0.59, 'unknown': 0.36}, n=98,
                        labels={'known_1': 'won first gamble', 'known_2': 'lost first gamble', 'act': 'play again'},
                        source='Tversky and Shafir (1992), Psychological Science 3(5)')
#: Conjunction fallacy in the Linda problem.
LINDA = dict(conjunction_rated_more_probable=0.85,
             source='Tversky and Kahneman (1983), Psychological Review 90(4)')


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
    >>> from quantum_mind.data import TWO_STAGE_GAMBLE, proportions_to_counts
    >>> proportions_to_counts(TWO_STAGE_GAMBLE['conditions'], 98)['unknown'].tolist()
    [35, 63]
    """
    return {k: np.array([int(round(n * p)), n - int(round(n * p))]) for k, p in props.items()}
