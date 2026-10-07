"""Order effects in human or LLM judgements
========================================

AI evaluation: order effects in a language model's or a human rater's yes/no judgements.

Workflow for any judge (people, crowd workers, LLMs): ask two questions in both orders across many
items or prompts, count the answer pairs, run the QQ test, and compare models. The counts below are
synthetic placeholders; replace them with your own.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_05.png'
import numpy as np
from quantum_mind.core import compare
from quantum_mind.families.order_effects import QuantumOrderModel, QuantumOrderModel4D, AnchoringOrderModel, BayesOrderModel, RANK_STRUCTURES, qq_test, rates

# %%
# Answer counts
# -------------
# Two yes/no judgements asked in both orders (synthetic counts; replace them with the counts from a
# human study or from a language model evaluated in both orders).
counts = {'AB': np.array([212, 48, 61, 179]),     # A asked first: yes-yes, yes-no, no-yes, no-no  (synthetic)
          'BA': np.array([240, 33, 52, 175])}     # B asked first  (synthetic)

# %%
# Test and compare
# ----------------
# The QQ test is parameter-free; the BIC comparison shows which model describes the order effect.
q, z, p = qq_test(counts)
print('QQ test: q = %.3f, z = %.2f, p = %.3f (p > 0.05: no evidence against the quantum-like regularity)' % (q, z, p))
for fr in compare([QuantumOrderModel4D, (QuantumOrderModel, {"structures": RANK_STRUCTURES}), AnchoringOrderModel, BayesOrderModel], counts, restarts=12):
    print('  %-22s BIC %.1f  rates %s' % (type(fr.model).__name__, fr.bic, rates(fr.model.predict()).round(3)))
