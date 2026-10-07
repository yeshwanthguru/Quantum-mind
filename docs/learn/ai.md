# Artificial intelligence

Artificial intelligence (AI) is the design of **agents** that choose actions to achieve goals. An
agent receives **percepts**, keeps some internal state, and picks the action that it expects to work
best. Machine learning, deep learning and reinforcement learning are ways of building parts of such
an agent from data instead of by hand.

```{mermaid}
flowchart LR
    accTitle: Artificial intelligence, diagram 1
    accDescr: An agent receives percepts from its environment, keeps a belief, weighs utilities and chooses the action with the highest expected utility.
    E(("environment")):::hum -->|percepts| A
    subgraph A["agent"]
      direction TB
      B["belief / state<br/>what is true now"]:::op --> U["utility<br/>what is good"]:::op --> C["choice<br/>maximise expected utility"]:::out
    end
    A -->|actions| E
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef hum fill:#3d1414,stroke:#ff7b72,color:#e6edf3
```

```{mermaid}
flowchart TB
    accTitle: Artificial intelligence, diagram 2
    accDescr: Artificial intelligence contains search, reasoning and machine learning; machine learning contains deep learning and reinforcement learning; embodied AI combines them in agents with bodies.
    AI["Artificial intelligence<br/>agents that act well"]:::op --> SR["search and planning"]:::in
    AI --> KR["knowledge and reasoning<br/>logic · probability"]:::in
    AI --> ML["machine learning<br/>learn from data"]:::in
    ML --> DL["deep learning<br/>many-layer networks"]:::out
    ML --> RL["reinforcement learning<br/>learn from reward"]:::out
    AI --> EAI["embodied AI<br/>agents with bodies"]:::out
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
```

## From the basics

Agent and environment
: The agent is the decision maker; the environment is everything else. A robot, a chess program and a
  chatbot are all agents in this sense.

Search
: Exploring sequences of actions to find one that reaches a goal (breadth-first, A\*, Monte Carlo tree
  search).

Knowledge representation and reasoning
: Writing down what is known (facts, rules, probabilities) so that new conclusions can be derived.
  **Bayesian networks** represent uncertain knowledge as a graph of variables and conditional
  probabilities.

Probability and Bayes' rule
: The standard way to update beliefs: $P(h \mid d) \propto P(d \mid h)\,P(h)$. Quantum Mind compares
  this classical rule with quantum probability, where the order of observations can matter.

Decision theory
: Choosing the action with the highest **expected utility**. When outcomes depend on what the agent
  does next, the problem becomes a **Markov decision process** (MDP), or a POMDP when the state is
  only partly observed.

Value of information
: How much a question or a sensor reading is worth: the expected improvement in the final decision
  minus the cost of getting it. A robot should ask a person only when the answer is worth more than the
  interruption.

Large language and vision-language models (LLMs, VLMs)
: Neural networks trained on text (and images) that can follow instructions. In robotics they are
  used to turn requests into plans; their answers can also depend on the order of the prompt.

## A small example: expected utility and the value of a question

A robot can pick up the red cup (it believes this is wanted with probability 0.7) or ask first.
Acting wrongly costs 5, asking costs 0.5, and the answer removes the doubt.

```python
p = 0.7
cost_act = 5 * (1 - p)        # expected cost of acting now
cost_ask = 0.5                # the answer makes the action right
print(cost_act, cost_ask)     # 1.5 > 0.5, so asking is worth it
```

The same rule, with the robot's uncertainty about its own models added, is
`quantum_mind.applications.robotics.ask_or_act`.

## Where Quantum Mind fits

Quantum Mind supplies the **belief** and **choice** parts of an agent that works with people:
quantum-like and classical models of how people answer, an ensemble that weighs them by evidence,
and decision rules (ask or act, question planning, hand-over) that use their predictions.

- [Tutorial: introduction](../tutorials/01_introduction.ipynb)
- [Tutorial: planning which question to ask](../tutorials/07_question_planner.ipynb)
- [Model atlas](../atlas/index.md)

## References

- Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
- Kochenderfer, M. J., Wheeler, T. A., & Wray, K. H. (2022). *Algorithms for Decision Making*. MIT Press.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.
- Howard, R. A. (1966). Information value theory. *IEEE Transactions on Systems Science and
  Cybernetics*, 2(1), 22-26.
