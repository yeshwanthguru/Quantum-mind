# A tour of the judgement families

Tutorials 2, 3 and 9 cover order effects, trust and perception. This tutorial walks through the other
quantum-like families, one short section each: what the effect is, the quantum-like model, the
classical baselines, and a fit or a check. Data are the published aggregates in `quantum_mind.data`
where they exist and are otherwise **simulated** (labelled as such).

```python
import itertools
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_mind.core import compare
theme = use_mpl_style('dark')
summary = {}            # effect sizes collected for the final figure
```

## 1. Conjunction fallacy

"Linda is a bank teller and a feminist" is judged more likely than "Linda is a bank teller" by most
people (85% in Tversky and Kahneman's study). The quantum-like model judges the conjunction by
thinking of the likelier event first; the classical joint model can never put the conjunction above
a single event.

```python
from quantum_mind.data import LINDA
from quantum_mind.families.conjunction import QuantumConjunctionModel, ClassicalJointModel, fallacy_rate
print(LINDA)
design = ['A', 'B', 'A&B', 'A|B']        # A = feminist (likely), B = bank teller (unlikely)
q = QuantumConjunctionModel(a=0.5, b=1.35, g=0.6).predict(design)
c = ClassicalJointModel(pA=0.8, pB=0.1, rho=0.5).predict(design)
for name, pred in (('quantum-like', q), ('classical joint', c)):
    print('%-16s P(A)=%.2f P(B)=%.2f P(A and B)=%.2f  fallacy: %s'
          % (name, pred['A'][0], pred['B'][0], pred['A&B'][0], fallacy_rate(pred)['conjunction_fallacy']))
summary['conjunction: P(A and B) - P(B)'] = q['A&B'][0] - q['B'][0]
```

## 2. Interference: the two-stage gamble

People who would play a second gamble after winning the first, and also after losing it, often
decline when they do not know the outcome. That violates the law of total probability.

```python
from quantum_mind.data import TWO_STAGE_GAMBLE, proportions_to_counts
from quantum_mind.families.interference import InterferenceModel, ClassicalMixtureModel, total_probability_bounds
obs = TWO_STAGE_GAMBLE['conditions']
lo, hi = total_probability_bounds(obs['known_1'], obs['known_2'])
print('observed', obs, ' classical range for "unknown": [%.2f, %.2f]' % (lo, hi))
counts = proportions_to_counts(obs, TWO_STAGE_GAMBLE['n'])
for r in compare([InterferenceModel, ClassicalMixtureModel], counts, restarts=10):
    print('  %-22s k=%d  BIC %.1f  P(play | unknown) = %.3f' % (type(r.model).__name__, r.k, r.bic, r.model.predict()['unknown'][0]))
summary['interference: shortfall below the classical range'] = lo - obs['unknown']
```

The interference model has four parameters for three proportions, so it fits exactly: it shows the
effect can be represented, not that it is predicted.

## 3. Quantum-like Bayesian network

Summing amplitudes over a hidden cause adds interference terms with phases.

```python
from quantum_mind.families.qlbn import BayesNet, classical_marginal, quantum_like_marginal
net = BayesNet({'First': ['win', 'lose'], 'Play': ['yes', 'no']}, {'Play': ['First']},
               {'First': {(): {'win': 0.5, 'lose': 0.5}},
                'Play': {('win',): {'yes': obs['known_1'], 'no': 1 - obs['known_1']},
                         ('lose',): {'yes': obs['known_2'], 'no': 1 - obs['known_2']}}})
phases = np.linspace(0, np.pi, 50)
ql = [quantum_like_marginal(net, 'Play', None, [0, th])['yes'] for th in phases]
print('classical P(play) = %.3f; quantum-like ranges %.3f to %.3f over the phase'
      % (classical_marginal(net, 'Play')['yes'], min(ql), max(ql)))
fig, ax = plt.subplots(figsize=(6.0, 2.8))
ax.plot(phases, ql, color=theme['palette'][2], lw=2.2, label='quantum-like')
ax.axhline(classical_marginal(net, 'Play')['yes'], color=theme['text'], ls='--', label='classical')
ax.axhline(obs['unknown'], color=theme['palette'][0], ls=':', label='observed (unknown outcome)')
ax.set_xlabel('phase θ'); ax.set_ylabel('P(play)'); ax.legend(); ax.set_title('Interference over a hidden cause')
plt.show()
```

## 4. Risky decisions: quantum decision theory

Choice probability = utility factor + attraction factor (uncertainty aversion). Compared with expected
utility and prospect theory on **simulated** choices.

```python
from quantum_mind.families.decision import QDTModel, ExpectedUtilityModel, ProspectTheoryModel
problems = {'gain_small': ([(30, 1.0)], [(45, 0.8), (0, 0.2)]), 'gain_large': ([(300, 1.0)], [(450, 0.8), (0, 0.2)]),
            'gain_long': ([(5, 1.0)], [(500, 0.01), (0, 0.99)]), 'loss_small': ([(-30, 1.0)], [(-45, 0.8), (0, 0.2)]),
            'mixed': ([(0, 1.0)], [(100, 0.5), (-80, 0.5)])}
data = QDTModel(alpha=0.85, beta=0.08, q0=0.2).sample(problems, 150, np.random.default_rng(5))
ranking = compare([QDTModel, ExpectedUtilityModel, ProspectTheoryModel], data, problems, restarts=6)
for r in ranking:
    print('  %-22s k=%d BIC %.1f' % (type(r.model).__name__, r.k, r.bic))
```

The choices were simulated from the QDT model, so its lower BIC is expected; the point is that the
three models can be told apart with 150 choices per problem.

## 5. Asymmetric similarity

"Korea is like China" is rated higher than "China is like Korea". In the quantum model the concept
with more knowledge has a larger subspace.

```python
from quantum_mind.families.similarity import QuantumSimilarityModel, GeometricModel, for_concepts, asymmetry
concepts = ['Korea', 'China', 'Japan']
pairs = list(itertools.permutations(concepts, 2))
Qs = for_concepts(QuantumSimilarityModel, concepts)
sim = Qs(concepts=concepts, ranks={'China': 2}, t0=0.6, p0=0.1, t1=0.9, p1=0.5, t2=1.1, p2=1.3).predict(pairs)
geo = for_concepts(GeometricModel, concepts)(concepts=concepts).predict(pairs)
print('quantum: Sim(Korea, China) - Sim(China, Korea) = %.3f' % asymmetry(sim, 'Korea', 'China'))
print('geometric (symmetric):                          %.3f' % asymmetry(geo, 'Korea', 'China'))
summary['similarity: asymmetry'] = asymmetry(sim, 'Korea', 'China')
```

## 6. Memory overdistribution and concept combination

```python
from quantum_mind.families.memory import QuantumEpisodicModel, overdistribution
from quantum_mind.families.concepts import FockSpaceConceptModel, MinConceptModel, overextension
mem = QuantumEpisodicModel(c=0.9, a_target=0.6, b_target=0.9, a_related=1.0, b_related=0.6, a_unrelated=1.3, b_unrelated=1.2)
od = {p: overdistribution(mem.predict(), p) for p in mem.PROBES}
print('memory overdistribution P(V)+P(G)-P(V or G):', {k: round(v, 3) for k, v in od.items()})
items = {'guppy': (0.30, 0.25), 'goldfish': (0.75, 0.90), 'shark': (0.05, 0.90)}
fock = FockSpaceConceptModel(m2=0.4, kappa=0.5).predict(items)
mini = MinConceptModel().predict(items)
print('"pet fish" membership, Fock space:', {k: round(float(v[0]), 2) for k, v in fock.items()})
print('"pet fish" membership, minimum rule:', {k: round(float(v[0]), 2) for k, v in mini.items()})
summary['memory: overdistribution (target)'] = od['target']
summary['concepts: overextension (guppy)'] = overextension(items, fock)['guppy']
```

## 7. Contextuality and quantum games

```python
from quantum_mind.families.contextuality import chsh, qubit_chsh_correlations
from quantum_mind.families.game_theory import EWLGame, PRISONERS_DILEMMA, C, D, Q
S = chsh(**qubit_chsh_correlations())
print('CHSH value of a Bell pair: %.3f (classical bound 2, quantum bound %.3f)' % (S, 2 * np.sqrt(2)))
G = EWLGame(PRISONERS_DILEMMA, np.pi / 2)
print("Prisoner's Dilemma, maximal entanglement: (Q,Q) payoff %s, Nash: %s" % (tuple(round(x, 2) for x in G.payoff(Q, Q)), G.is_nash(Q, Q, grid=21)))
print("classical game (gamma = 0): (D,D) payoff %s" % (EWLGame(PRISONERS_DILEMMA, 0.0).payoff(D, D),))
summary['CHSH: value above 2'] = S - 2
```

## 8. Effects at a glance

```python
fig, ax = plt.subplots(figsize=(7.0, 3.0))
names = list(summary)
ax.barh(names[::-1], [summary[k] for k in names][::-1], color=theme['palette'][:len(names)])
ax.axvline(0, color=theme['text'], lw=1)
ax.set_xlabel('size of the effect (0 = classical prediction)'); ax.set_title('Effects classical probability cannot produce')
plt.show()
```

The statements below are checked by this cell every time the documentation is built:

```python
assert q['A&B'][0] > q['B'][0] and c['A&B'][0] <= min(c['A'][0], c['B'][0])     # fallacy only in the QL model
assert obs['unknown'] < lo                                                     # the data violate the classical law
assert min(ql) < obs['unknown'] < max(ql)                                      # a phase reproduces the data
assert abs(asymmetry(geo, 'Korea', 'China')) < 1e-12 < abs(summary['similarity: asymmetry'])
assert S > 2 and od['target'] > 0 and summary['concepts: overextension (guppy)'] > 0
print('checked')
```

**Summary.** Each family produces an effect that its classical baselines cannot: a conjunction above a
single event, a violation of the law of total probability, asymmetric similarity, memory
overdistribution, concept overextension and correlations above the CHSH bound. Producing an effect is
not the same as predicting new data; tutorial 2 shows how to test a model's distinctive predictions.

References: Tversky & Kahneman (1983); Tversky & Shafir (1992); Busemeyer et al. (2011); Moreira &
Wichert (2016); Yukalov & Sornette (2011); Pothos, Busemeyer & Trueblood (2013); Brainerd, Wang & Reyna
(2013); Aerts (2009); Clauser, Horne, Shimony & Holt (1969); Eisert, Wilkens & Lewenstein (1999).
