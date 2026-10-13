"""Bridge between the small robot language model and a local general-purpose LLM, with the LLM kept on a short leash.

A general model (for example Llama or Qwen served by Ollama on your own machine) understands free language the small model never
saw. The bridge uses it in two limited ways and checks everything it says:

* :func:`normalise` asks the LLM to rewrite a free-form request into the robot's own 32-word vocabulary; the reply is accepted only if every
  word is in the vocabulary, and then the small model grounds it.
* :func:`llm_parse` asks the LLM for the final referent as a small JSON object (``red``, ``green``, ``blue``, ``leftmost``, ``rightmost`` or
  ``none``); anything else is rejected.

:class:`LLMBridge` compares the small model's belief with the LLM's reading. When they disagree, or when the LLM's reply is
unusable, the robot should ask the person rather than trust either (``result.ask``). If the LLM is unavailable the small model
works alone.

``generate`` is any ``str -> str`` function, for example :func:`quantum_mind.embodied.ollama_generate`. The test suite uses
scripted replies only: no language model was run while this was written."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from . import corpus as C

__all__ = ['normalise', 'llm_parse', 'belief_from_parse', 'LLMBridge', 'BridgeResult']

REFERENCES = C.COLORS + ('leftmost', 'rightmost', 'none')


def _json(text):
    m = re.search(r'\{.*\}', text, re.S)
    return json.loads(m.group(0)) if m else None


def normalise(text, generate):
    """Rewrite ``text`` into the robot vocabulary with the LLM; ``None`` if the reply uses any other word.

    The instruction to the LLM lists the allowed words and asks for one sentence, nothing else.
    """
    allowed = [w for w in C.WORDS if not w.startswith('<')]
    prompt = ('Rewrite the robot instruction below as ONE short sentence using only these words: %s. If the speaker corrects '
              'themselves, keep the correction. Reply with the sentence only.\nInstruction: %s' % (', '.join(allowed), text))
    try:
        reply = generate(prompt)
    except Exception:
        return None
    sentence = ' '.join(re.findall(r"[a-z]+", str(reply).lower()))
    return sentence if sentence and all(w in C.VOCAB for w in sentence.split()) else None


def llm_parse(text, generate):
    """The LLM's reading of the final referent: ``'red'``, ``'green'``, ``'blue'``, ``'leftmost'``, ``'rightmost'`` or ``'none'``.

    Returns ``None`` when the reply is not a JSON object with an allowed ``reference``.
    """
    prompt = ('A robot is told to go to an object. Give the FINAL object the speaker means (after any correction) as JSON '
              '{"reference": one of %s}. Reply with the JSON only.\nInstruction: %s' % (', '.join(REFERENCES), text))
    try:
        out = _json(str(generate(prompt)))
    except Exception:
        return None
    ref = out.get('reference') if isinstance(out, dict) else None
    return ref if ref in REFERENCES else None


def belief_from_parse(reference, cells, colors):
    """Belief over the 16 cells implied by a parsed reference (uniform over the matching objects), or ``None`` for ``'none'``/invalid."""
    if reference in (None, 'none'):
        return None
    if reference in C.COLORS:
        hit = [c for c, col in zip(cells, colors) if C.COLORS[col] == reference]
    else:
        hit = [cells[0] if reference == 'leftmost' else cells[-1]]
    if not hit:
        return None
    b = np.zeros(C.N_CELLS)
    b[hit] = 1.0 / len(hit)
    return b


@dataclass
class BridgeResult:
    """What the bridge decided.

    Attributes
    ----------
    belief : numpy.ndarray
        Belief over the 16 cells (the small model's, tempered by the LLM's reading when both exist).
    lm_belief, llm_belief : numpy.ndarray or None
    sentence : str or None
        The sentence the small model read (the original if it was already in vocabulary, else the LLM's rewrite).
    ask : bool
        True when the robot should ask the person: no usable reading, or the two readings disagree.
    reason : str
    """
    belief: np.ndarray
    lm_belief: np.ndarray
    llm_belief: np.ndarray
    sentence: str
    ask: bool
    reason: str


class LLMBridge:
    """Ground free-form language with the small model, checked by a local LLM.

    Parameters
    ----------
    lm : RobotLanguageModel
    generate : callable or None
        ``str -> str`` text generator for the local LLM; ``None`` uses the small model alone.
    disagreement : float, optional
        Probability mass the two readings may differ by before the bridge asks (total variation distance).
    """

    def __init__(self, lm, generate=None, disagreement=0.5):
        self.lm, self.generate, self.disagreement = lm, generate, disagreement
        self.stats = dict(calls=0, rewrites=0, rejected_rewrites=0, rejected_parses=0, disagreements=0)

    def ground(self, text, image, cells, colors):
        """Ground ``text`` for a scene given its image strip and the object cells and colours (left to right)."""
        sentence = text if all(w in C.VOCAB for w in text.split()) else None
        if sentence is None and self.generate is not None:
            self.stats['calls'] += 1
            sentence = normalise(text, self.generate)
            self.stats['rewrites' if sentence else 'rejected_rewrites'] += 1
        if sentence is None:
            return BridgeResult(np.full(C.N_CELLS, 1 / C.N_CELLS), None, None, None, True,
                                'the request could not be put in the robot vocabulary')
        lm_b = self.lm.ground_text(sentence, image)
        llm_b = None
        if self.generate is not None:
            ref = llm_parse(text, self.generate)
            llm_b = belief_from_parse(ref, cells, colors)
            if llm_b is None:
                self.stats['rejected_parses'] += 1
        if llm_b is None:
            return BridgeResult(lm_b, lm_b, None, sentence, False, 'small model alone')
        tv = 0.5 * np.abs(lm_b - llm_b).sum()
        if tv > self.disagreement:
            self.stats['disagreements'] += 1
            return BridgeResult(lm_b, lm_b, llm_b, sentence, True, 'the small model and the LLM disagree (total variation %.2f)' % tv)
        fused = lm_b * (llm_b + 0.02)
        return BridgeResult(fused / fused.sum(), lm_b, llm_b, sentence, False, 'the two readings agree')
