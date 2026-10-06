r"""Adaptive questioning: which question a robot should ask next, and when to stop asking.

The robot is unsure which of several hypotheses holds (which object the person means, which plan they
prefer). It can ask yes/no questions, each with a cost, or act on its best hypothesis, risking an
error. After each answer it updates its belief and decides again.

The answer model matters. People's answers depend on the questions asked before (order effects), so a
planner that treats answers as independent readings of a fixed preference draws wrong conclusions from
the second and later answers. Two answer models are provided:

* :class:`ProjectiveAnswerModel`: each hypothesis is a belief state and each question a projector;
  answers follow the Lüders rule in sequence, so they depend on what was asked before;
* :class:`IndependentAnswerModel`: a fixed table of yes-probabilities per hypothesis and question
  (the classical, order-free assumption).

:class:`QuestionPlanner` works with either. At each step it computes the *value of information* of every
remaining question, the expected reduction in the cost of acting wrongly,

.. math:: \text{VOI}(q) = C_{\text{err}}\,\bigl(1 - \max_h p(h)\bigr)
          - \sum_a p(a \mid q)\, C_{\text{err}}\,\bigl(1 - \max_h p(h \mid a)\bigr),

asks the question with the largest value when it exceeds the cost of asking, and acts otherwise.

Examples
--------
>>> import numpy as np
>>> from quantum_mind.applications.questioning import IndependentAnswerModel, QuestionPlanner
>>> m = IndependentAnswerModel(['red cup', 'blue cup'], ['red?'], [[0.95], [0.05]])
>>> planner = QuestionPlanner(m, ask_cost=0.2, error_cost=5.0)
>>> planner.decide([])[:2]                 # 50/50: worth asking
('ask', 'red?')
>>> planner.decide([('red?', 1)])[:2]      # after "yes": act on the red cup
('act', 'red cup')
"""
from __future__ import annotations

import numpy as np
from ..core.linalg import normalize

__all__ = ['ProjectiveAnswerModel', 'IndependentAnswerModel', 'QuestionPlanner']


class ProjectiveAnswerModel:
    """Order-dependent answers: Lüders rule on a belief state per hypothesis.

    Parameters
    ----------
    states : dict
        ``{hypothesis: state vector}``: the person's belief state when the hypothesis is true.
    projectors : dict
        ``{question: projector}`` of the "yes" answer.
    """

    def __init__(self, states, projectors):
        self.hypotheses = list(states)
        self.questions = list(projectors)
        self.states = {h: normalize(v) for h, v in states.items()}
        self.projectors = projectors

    def _state_after(self, h, history):
        """Probability of the history and the post-answer state, for hypothesis h."""
        psi = self.states[h]
        prob = 1.0
        for q, a in history:
            P = self.projectors[q] if a else np.eye(len(psi)) - self.projectors[q]
            w = P @ psi
            p = float(np.real(np.vdot(w, w)))
            prob *= p
            psi = w / np.sqrt(p) if p > 1e-15 else w
        return prob, psi

    def likelihood(self, h, history):
        """Probability of the whole answer history under one hypothesis.

        Parameters
        ----------
        h : hashable
            Hypothesis.
        history : list of tuple
            ``[(question, answer), ...]`` with answer 1 (yes) or 0 (no), in the order asked.

        Returns
        -------
        float
        """
        return self._state_after(h, history)[0]

    def p_yes(self, h, history, q):
        """Probability of "yes" to the next question, given the history.

        Parameters
        ----------
        h : hashable
        history : list of tuple
        q : hashable
            Next question.

        Returns
        -------
        float
        """
        _, psi = self._state_after(h, history)
        w = self.projectors[q] @ psi
        return float(np.real(np.vdot(w, w)))

    def order_free(self):
        """The order-blind approximation of this model.

        Returns
        -------
        IndependentAnswerModel
            Yes-probabilities of every question asked first, with answers treated as independent.
        """
        table = [[self.p_yes(h, [], q) for q in self.questions] for h in self.hypotheses]
        return IndependentAnswerModel(self.hypotheses, self.questions, table)


class IndependentAnswerModel:
    """Order-free answers: a fixed yes-probability per hypothesis and question.

    Parameters
    ----------
    hypotheses : sequence
        Hypothesis names.
    questions : sequence
        Question names.
    table : array_like
        ``table[i][j]`` = P(yes to question j | hypothesis i).
    """

    def __init__(self, hypotheses, questions, table):
        self.hypotheses, self.questions = list(hypotheses), list(questions)
        self.table = np.asarray(table, float)
        if self.table.shape != (len(self.hypotheses), len(self.questions)):
            raise ValueError('table must have one row per hypothesis and one column per question')

    def likelihood(self, h, history):
        """Probability of the answer history under one hypothesis (answers independent).

        Parameters
        ----------
        h : hashable
        history : list of tuple

        Returns
        -------
        float
        """
        i = self.hypotheses.index(h)
        out = 1.0
        for q, a in history:
            p = self.table[i, self.questions.index(q)]
            out *= p if a else 1 - p
        return out

    def p_yes(self, h, history, q):
        """Probability of "yes" to a question (independent of the history).

        Parameters
        ----------
        h : hashable
        history : list of tuple
        q : hashable

        Returns
        -------
        float
        """
        return float(self.table[self.hypotheses.index(h), self.questions.index(q)])


class QuestionPlanner:
    """Value-of-information question planner.

    Parameters
    ----------
    answer_model : ProjectiveAnswerModel or IndependentAnswerModel
        How the robot believes people answer.
    prior : array_like, optional
        Prior over hypotheses (uniform by default).
    ask_cost : float, optional
        Cost of asking one question (time, annoyance).
    error_cost : float, optional
        Cost of acting on a wrong hypothesis.
    repeat : bool, optional
        Allow asking the same question twice (default False).
    """

    def __init__(self, answer_model, prior=None, ask_cost=1.0, error_cost=5.0, repeat=False):
        self.model = answer_model
        H = len(answer_model.hypotheses)
        self.prior = np.full(H, 1 / H) if prior is None else np.asarray(prior, float) / np.sum(prior)
        self.ask_cost, self.error_cost, self.repeat = ask_cost, error_cost, repeat

    def posterior(self, history):
        """Belief over hypotheses after an answer history.

        Parameters
        ----------
        history : list of tuple
            ``[(question, answer), ...]``.

        Returns
        -------
        numpy.ndarray
        """
        lik = np.array([self.model.likelihood(h, history) for h in self.model.hypotheses])
        post = self.prior * lik
        s = post.sum()
        return post / s if s > 0 else np.full(len(post), 1 / len(post))

    def _expected_error(self, post):
        """Expected cost of acting now on the most probable hypothesis."""
        return self.error_cost * (1 - post.max())

    def value_of_information(self, q, history):
        """Expected reduction of the error cost from asking one more question.

        Parameters
        ----------
        q : hashable
            Candidate question.
        history : list of tuple

        Returns
        -------
        float
        """
        post = self.posterior(history)
        p_yes_h = np.array([self.model.p_yes(h, history, q) for h in self.model.hypotheses])
        p_yes = float(post @ p_yes_h)
        after = 0.0
        for a, pa in ((1, p_yes), (0, 1 - p_yes)):
            if pa > 1e-12:
                after += pa * self._expected_error(self.posterior(history + [(q, a)]))
        return self._expected_error(post) - after

    def information_gain(self, q, history):
        """Expected entropy reduction (bits) of the belief from asking a question.

        Parameters
        ----------
        q : hashable
        history : list of tuple

        Returns
        -------
        float
        """
        def H(p):
            p = p[p > 0]
            return float(-np.sum(p * np.log2(p)))
        post = self.posterior(history)
        p_yes = float(post @ np.array([self.model.p_yes(h, history, q) for h in self.model.hypotheses]))
        return H(post) - sum(pa * H(self.posterior(history + [(q, a)]))
                             for a, pa in ((1, p_yes), (0, 1 - p_yes)) if pa > 1e-12)

    def decide(self, history):
        """Ask or act.

        Parameters
        ----------
        history : list of tuple
            Answers so far.

        Returns
        -------
        tuple
            ``('ask', question, value_of_information)`` or ``('act', hypothesis, probability)``.
        """
        asked = {q for q, _ in history}
        options = [q for q in self.model.questions if self.repeat or q not in asked]
        post = self.posterior(history)
        if options:
            vois = [self.value_of_information(q, history) for q in options]
            k = int(np.argmax(vois))
            if vois[k] > self.ask_cost:
                return 'ask', options[k], float(vois[k])
        i = int(np.argmax(post))
        return 'act', self.model.hypotheses[i], float(post[i])

    def run(self, respond, max_questions=10):
        """Interactive loop: ask until acting is better.

        Parameters
        ----------
        respond : callable
            ``respond(question) -> 0 or 1``: asks the person (or a simulated person).
        max_questions : int, optional
            Safety limit on the number of questions.

        Returns
        -------
        dict
            ``choice`` (hypothesis acted on), ``probability``, ``history`` and ``cost`` (asking cost only).
        """
        history = []
        for _ in range(max_questions):
            d = self.decide(history)
            if d[0] == 'act':
                break
            history.append((d[1], int(respond(d[1]))))
        d = self.decide(history) if not history or d[0] == 'ask' else d
        if d[0] == 'ask':                       # question limit reached: act on the best hypothesis
            post = self.posterior(history)
            i = int(np.argmax(post))
            d = ('act', self.model.hypotheses[i], float(post[i]))
        return {'choice': d[1], 'probability': d[2], 'history': history, 'cost': self.ask_cost * len(history)}

    def simulate(self, truth_model, true_h, rng, max_questions=10):
        """Run the planner against a simulated person.

        Parameters
        ----------
        truth_model : ProjectiveAnswerModel or IndependentAnswerModel
            How the simulated person really answers (may differ from the planner's own model).
        true_h : hashable
            The hypothesis that is actually true.
        rng : numpy.random.Generator
        max_questions : int, optional

        Returns
        -------
        dict
            As :meth:`run`, plus ``correct`` and ``total_cost`` (asking plus error cost).
        """
        history = []

        def respond(q):
            return int(rng.random() < truth_model.p_yes(true_h, history, q))

        def tracked(q):
            a = respond(q)
            history.append((q, a))
            return a
        out = self.run(tracked, max_questions)
        out['correct'] = out['choice'] == true_h
        out['total_cost'] = out['cost'] + (0 if out['correct'] else self.error_cost)
        return out
