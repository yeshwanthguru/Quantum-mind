"""Marketing, linguistics, perception: asymmetric similarity ('Korea is like China' rated higher than
'China is like Korea'). Ratings are simulated from the quantum similarity model; the quantum model, a
biased geometric model (Nosofsky, 1991) and a symmetric geometric model are fitted."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import itertools
import numpy as np
from qlcog.core import compare
from qlcog.families.similarity import QuantumSimilarityModel, GeometricModel, BiasedGeometricModel, for_concepts, asymmetry

concepts = ['Korea', 'China', 'Japan']
pairs = [(a, b) for a, b in itertools.permutations(concepts, 2)]
Q = for_concepts(QuantumSimilarityModel, concepts)
truth = Q(concepts=concepts, ranks={'China': 2}, t0=0.6, p0=0.1, t1=0.9, p1=0.5, t2=1.1, p2=1.3)
pred = truth.predict(pairs)
print('Sim(Korea, China) - Sim(China, Korea) =', round(asymmetry(pred, 'Korea', 'China'), 3))
rng = np.random.default_rng(4)
data = {k: np.clip(v + rng.normal(0, 0.01, 1), 0, 1) for k, v in pred.items()}     # synthetic mean ratings
for fr in compare([(Q, dict(concepts=concepts, ranks={'China': 2})), (for_concepts(BiasedGeometricModel, concepts), dict(concepts=concepts)),
                   (for_concepts(GeometricModel, concepts), dict(concepts=concepts))], data, pairs, restarts=8):
    print('  %-22s k=%2d SSE %.5f BIC %.1f' % (fr.model.name, fr.k, fr.loss, fr.bic))
