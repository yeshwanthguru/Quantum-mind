"""AI evaluation: order effects in a language model's or a human rater's yes/no judgements.

Workflow for any judge (people, crowd workers, LLMs): ask two questions in both orders across many
items or prompts, count the answer pairs, run the QQ test, and compare models. The counts below are
synthetic placeholders; replace them with your own."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from qlcog.core import compare
from qlcog.families.order_effects import QuantumOrderModel, QuantumOrderModel4D, AnchoringOrderModel, BayesOrderModel, RANK_STRUCTURES, qq_test, rates

counts = {'AB': np.array([212, 48, 61, 179]),     # A asked first: yes-yes, yes-no, no-yes, no-no  (synthetic)
          'BA': np.array([240, 33, 52, 175])}     # B asked first  (synthetic)
q, z, p = qq_test(counts)
print('QQ test: q = %.3f, z = %.2f, p = %.3f (p > 0.05: no evidence against the quantum-like regularity)' % (q, z, p))
for fr in compare([QuantumOrderModel4D, (QuantumOrderModel, {"structures": RANK_STRUCTURES}), AnchoringOrderModel, BayesOrderModel], counts, restarts=12):
    print('  %-22s BIC %.1f  rates %s' % (type(fr.model).__name__, fr.bic, rates(fr.model.predict()).round(3)))
