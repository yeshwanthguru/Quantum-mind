"""Information retrieval: ranking documents with a quantum language model (density matrices with term
dependencies) versus unigram query likelihood, on a tiny illustrative corpus. The two models agree on
the top document for both queries and differ lower down; with five documents this shows how the
models are used, not which one is better (that needs a retrieval benchmark)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
from qlcog.inspired import QuantumLanguageModel, QueryLikelihoodModel

docs = ['the robot picks up the red cup from the table',
        'a red apple, a green cup and a plate',
        'the cup is on the left and the red light is on',
        'red cup red cup: the robot found the red cup',
        'robot arms and conveyor belts']
for query in ('red cup', 'robot table'):
    for model in (QuantumLanguageModel(window=2), QueryLikelihoodModel()):
        model.fit(docs); order = model.rank(query)
        print('%-12s %-22s ranking %s' % (query, type(model).__name__, [int(i) for i in order]))
