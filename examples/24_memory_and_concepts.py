"""Memory and concepts (simulated judgements): two effects classical probability cannot produce.

1. Overdistribution in recognition memory: P(studied) + P(related) exceeds P(studied or related).
2. Overextension in concept combination (the guppy effect): membership in "A and B" exceeds the
   membership in A or in B.
Data are simulated from the quantum-like models with noise (illustrative values, not experimental
data); the classical baselines are fitted to the same data."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from qlcog.core import compare
from qlcog.families.memory import QuantumEpisodicModel, AdditiveMemoryModel, overdistribution
from qlcog.families.concepts import (FockSpaceConceptModel, ProductConceptModel, MinConceptModel,
                                     WeightedAverageModel, overextension)

rng = np.random.default_rng(0)
gen = QuantumEpisodicModel(c=0.9, a_target=0.6, b_target=0.9, a_related=1.0, b_related=0.6, a_unrelated=1.3, b_unrelated=1.2)
data = {k: np.clip(v + rng.normal(0, 0.02, 1), 0, 1) for k, v in gen.predict().items()}
print('overdistribution in the simulated data:', {p: round(overdistribution(data, p), 3) for p in gen.PROBES})
for r in compare([QuantumEpisodicModel, AdditiveMemoryModel], data):
    print('  %-26s SSE %.4f  BIC %.1f' % (r.model.name, r.loss, r.bic))

items = {'guppy': (0.30, 0.25), 'goldfish': (0.75, 0.90), 'shark': (0.05, 0.90), 'trout': (0.10, 0.80),
         'canary': (0.85, 0.02), 'tuna': (0.02, 0.95)}
truth = FockSpaceConceptModel(m2=0.4, kappa=0.5).predict(items)
obs = {k: np.clip(v + rng.normal(0, 0.02, 1), 0, 1) for k, v in truth.items()}
print('\noverextension "pet and fish" (simulated):', {k: round(v, 2) for k, v in overextension(items, obs).items()})
for r in compare([FockSpaceConceptModel, WeightedAverageModel, MinConceptModel, ProductConceptModel], obs, items):
    print('  %-22s SSE %.4f  BIC %.1f  %s' % (r.model.name, r.loss, r.bic, {k: round(v, 2) for k, v in r.model.params.items()}))
