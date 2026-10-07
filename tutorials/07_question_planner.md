# Planning which question to ask

The robot sees three objects (red cup, blue cup, red bowl) and hears "bring me that one". It can ask
yes/no questions ("is it red?", "is it a cup?", "is it on the left?") before acting. Which question
should it ask first, and when should it stop? The **question planner** computes the **value of
information** of each question: the expected reduction in the cost of the final decision, minus the
cost of asking.

The twist: people's answers depend on what was asked before. A **projective** answer model captures
this; an **independent** model (the classical default) ignores it. Both are compared on simulated
people.

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_mind.envs.hri import default_domain
from quantum_mind.applications.questioning import QuestionPlanner
theme = use_mpl_style('dark')
person = default_domain()          # a simulated person whose answers follow a projective model
print('hypotheses:', person.hypotheses, '   questions:', person.questions)
```

## 1. How informative is each question?

```python
planner = QuestionPlanner(person, ask_cost=0.3, error_cost=5.0)
voi = {q: planner.value_of_information(q, []) for q in person.questions}
gain = {q: planner.information_gain(q, []) for q in person.questions}
for q in person.questions:
    print('%-6s value of information %.3f   information gain %.3f bits' % (q, voi[q], gain[q]))
print('decision now:', planner.decide([]))
```

```python
fig, ax = plt.subplots(figsize=(6.0, 2.8))
x = np.arange(len(voi))
ax.bar(x - 0.18, list(voi.values()), 0.36, label='value of information', color=theme['palette'][1])
ax.bar(x + 0.18, list(gain.values()), 0.36, label='information gain (bits)', color=theme['palette'][2])
ax.set_xticks(x, list(voi)); ax.set_title('Which question first?'); ax.legend()
plt.show()
```

## 2. Order dependence

The probability of a "yes" to one question depends on the answer to an earlier one, and on the
order. The independent model cannot represent this.

```python
blind = person.order_free()
for h in person.hypotheses:
    a = person.likelihood(h, [('red?', 1), ('cup?', 1)])
    b = person.likelihood(h, [('cup?', 1), ('red?', 1)])
    print('%-9s P(yes, yes): red first %.3f, cup first %.3f   (independent model: %.3f both)'
          % (h, a, b, blind.likelihood(h, [('red?', 1), ('cup?', 1)])))
```

## 3. Does modelling the order help?

Run both planners against the same simulated people, many times, for each target object.

```python
rng = np.random.default_rng(0)
aware, naive = QuestionPlanner(person, ask_cost=0.3, error_cost=5.0), QuestionPlanner(blind, ask_cost=0.3, error_cost=5.0)
res = {'order-aware': [], 'independent': []}
for i in range(300):
    target = person.hypotheses[i % 3]
    for name, pl in (('order-aware', aware), ('independent', naive)):
        out = pl.simulate(person, target, np.random.default_rng(i))
        res[name].append((out['total_cost'], out['correct'], len(out['history'])))
for name, r in res.items():
    r = np.array(r, float)
    print('%-12s mean cost %.3f   accuracy %.3f   questions asked %.2f' % (name, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean()))
```

```python
fig, ax = plt.subplots(figsize=(6.0, 2.8))
for (name, r), c in zip(res.items(), theme['palette']):
    ax.hist(np.array(r)[:, 0], bins=np.arange(0, 6, 0.3), alpha=0.6, color=c, label=name)
ax.set_xlabel('total cost of the episode'); ax.set_ylabel('episodes'); ax.legend()
ax.set_title('Episode costs on simulated people')
plt.show()
```

In this domain the gain from modelling order is modest. Across 60 randomly drawn domains (see the
CHANGELOG for 2.0.0) the order-aware planner saved 0.18 cost units on average, with a standard
deviation of 0.21, and was better in about half of the domains: the benefit depends on how strongly
the questions interfere.

References: Howard (1966); Wang & Busemeyer (2013), *Topics in Cognitive Science* 5, 689-710.
