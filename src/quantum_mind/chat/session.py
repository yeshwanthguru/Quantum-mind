"""A conversation with a local LLM: history, streaming, and an optional "am I sure?" check.

:class:`ChatSession` keeps the messages of a conversation (a system prompt and the most recent turns) and streams the model's
answer. The optional :class:`ConsistencyCheck` asks the same question several more times at a higher temperature and measures
how much the answers agree (self-consistency); when they do not, the reply carries a note that the model is unsure. That is a
cheap signal, not a guarantee: a model can be consistently wrong, and a correct answer can be worded differently each time."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ['ChatSession', 'ConsistencyCheck', 'agreement']

DEFAULT_SYSTEM = ('You are a helpful, honest assistant running on the user\'s own computer. Answer clearly and briefly. '
                  'If you are not sure, or the question needs information you do not have, say so instead of guessing.')
STOP = frozenset('a an the is are was were be been of to in on at for and or it this that as by with from i you we they he she do does did '
                 'not no yes can will would should could may might has have had so if then than which who what when where'.split())


def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if w not in STOP]


def agreement(a, b):
    """Word-overlap agreement (F1 of the content words, stop words dropped) between two answers, in [0, 1].

    Answers that give different numbers are halved: ``"the answer is 42"`` and ``"the answer is 17"`` disagree even though most words match.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 1.0 if ta == tb else 0.0
    sa, sb = {}, {}
    for w in ta:
        sa[w] = sa.get(w, 0) + 1
    for w in tb:
        sb[w] = sb.get(w, 0) + 1
    common = sum(min(sa[w], sb[w]) for w in sa if w in sb)
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    f1 = 2 * p * r / (p + r)
    na, nb = {w for w in ta if w[0].isdigit()}, {w for w in tb if w[0].isdigit()}
    return f1 * 0.5 if (na and nb and not (na & nb)) else f1                  # two different numbers are a disagreement, not a rewording


@dataclass
class ConsistencyCheck:
    """Ask again ``n - 1`` times at a higher temperature and compare the answers with the first one.

    Parameters
    ----------
    n : int
        Total answers compared (the streamed one plus ``n - 1`` samples).
    temperature : float
        Sampling temperature of the extra answers.
    threshold : float
        Mean agreement below this adds a note to the reply.
    """
    n: int = 3
    temperature: float = 0.8
    threshold: float = 0.5
    last: float = field(default=None, init=False)

    def score(self, backend, messages, first, max_tokens=None):
        """Mean agreement of ``first`` with ``n - 1`` fresh samples (``None`` when ``n < 2``)."""
        if self.n < 2:
            return None
        scores = []
        for _ in range(self.n - 1):
            other = ''.join(backend.stream(messages, temperature=self.temperature, max_tokens=max_tokens))
            scores.append(agreement(first, other))
        self.last = sum(scores) / len(scores)
        return self.last


class ChatSession:
    """A conversation with a backend.

    Parameters
    ----------
    backend : OllamaBackend, LlamaCppBackend or ScriptedBackend
    system : str
        System prompt.
    max_turns : int
        Keep at most this many recent question-answer pairs (the oldest are dropped).
    check : ConsistencyCheck or None
        Optional self-consistency check on every answer.
    max_tokens : int or None
        Longest answer to ask for.
    """

    def __init__(self, backend, system=DEFAULT_SYSTEM, max_turns=20, check=None, max_tokens=None):
        self.backend, self.system, self.max_turns, self.check, self.max_tokens = backend, system, max_turns, check, max_tokens
        self.history = []
        self.last_agreement = None

    def reset(self):
        """Forget the conversation."""
        self.history = []
        self.last_agreement = None

    def messages(self, question=None):
        msgs = [{'role': 'system', 'content': self.system}] + self.history
        return msgs + [{'role': 'user', 'content': question}] if question is not None else msgs

    def ask(self, question):
        """Stream the answer to ``question`` piece by piece. The turn is added to the history when the stream ends or is closed.

        If a :class:`ConsistencyCheck` is set, a final piece may be a note that the answers varied (``session.last_agreement`` holds
        the score).
        """
        msgs = self.messages(question)
        parts = []
        try:
            for piece in self.backend.stream(msgs, max_tokens=self.max_tokens):
                parts.append(piece)
                yield piece
        finally:                                                           # an interrupted answer still joins the history
            answer = ''.join(parts).strip()
            if answer:
                self.history += [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
                self.history = self.history[-2 * self.max_turns:]
        self.last_agreement = None
        if self.check is not None and answer:
            self.last_agreement = self.check.score(self.backend, msgs, answer, self.max_tokens)
            if self.last_agreement is not None and self.last_agreement < self.check.threshold:
                yield '\n[not sure: repeated answers to this question agreed only %.0f%%; check it elsewhere]' % (100 * self.last_agreement)

    def answer(self, question):
        """The whole answer to ``question`` as a string."""
        return ''.join(self.ask(question)).strip()

    def generate(self, prompt, **options):
        """One-off answer with no history, so a session can be used as the ``generate`` function of the library's LLM bridges."""
        return ''.join(self.backend.stream([{'role': 'system', 'content': self.system}, {'role': 'user', 'content': prompt}], **options))
