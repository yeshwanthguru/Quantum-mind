# Quantum-like cognition

Quantum-like (or quantum cognition) models use the **mathematics** of quantum probability to describe
human judgement. Nothing in the brain is assumed to be quantum. The motivation is empirical: people's
answers show patterns that classical probability handles badly, such as **question-order effects**,
the **conjunction fallacy** and violations of the **law of total probability**. Quantum probability
represents beliefs as a vector and questions as **projections**, and projections need not commute, so
the order of questions can matter.

```{mermaid}
flowchart LR
    accTitle: Quantum-like cognition, diagram 1
    accDescr: A belief state is projected onto the yes subspace of question A, renormalised, then projected onto question B; the result is compared with the opposite order.
    B["belief state ψ<br/>unit vector"]:::hum --> PA["question A<br/>projector Pₐ"]:::op
    PA -->|"P(yes to A) = ‖Pₐψ‖²"| S1["state after answering A<br/>Pₐψ / ‖Pₐψ‖"]:::op
    S1 --> PB["question B<br/>projector P_b"]:::op --> R["P(yes A, then yes B)<br/>= ‖P_b Pₐ ψ‖²"]:::out
    R --> CMP{"compare with<br/>B-then-A order"}:::base
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef hum fill:#3d1414,stroke:#ff7b72,color:#e6edf3
    classDef base fill:#262626,stroke:#8b949e,color:#e6edf3,stroke-dasharray:4 3
```

## From the basics

Quantum-like versus quantum
: **Quantum-like** models borrow the probability rules and run on any computer. **Quantum** models run
  as circuits on quantum hardware. **Quantum-inspired** algorithms borrow quantum ideas for classical
  heuristics. The [concepts page](../user_guide/concepts.md) explains the difference.

Projection and Lüders' rule
: Answering "yes" projects the belief onto the "yes" subspace and renormalises it. If projectors do
  not commute, $\|P_b P_a \psi\|^2 \neq \|P_a P_b \psi\|^2$ in general: the order effect.

The QQ equality
: Projective models make a parameter-free prediction: the total probability of giving the **same**
  answer to both questions does not depend on the order. Wang and Busemeyer tested it on dozens of
  national surveys. Quantum Mind provides the test (`qq_test`).

Interference
: The probability of an outcome when an intermediate question is not asked can differ from the
  average over its answers. The difference is an interference term, as in the double-slit experiment.

Open-system dynamics
: Beliefs that drift and lose coherence over time are modelled with density matrices and the Lindblad
  equation. Quantum Mind uses this for **trust**, which a robot can change just by asking about it.

Fair comparison
: A quantum-like model is useful only if it predicts better than classical models fitted to the same
  data. Every family in the library ships with classical baselines and a distinctive test.

## A small example: an order effect in three lines

```python
from quantum_mind.families.order_effects import QuantumOrderModel
m = QuantumOrderModel(a=2.35, b=0.97, g=0.58, ranks=(1, 2))
p = m.predict()
print(p['AB'].round(3), p['BA'].round(3))   # the joint answers depend on the order
```

## Where Quantum Mind fits

This is the library's first pillar: twelve families of quantum-like models of judgement, decision,
memory, concepts, trust and perception, each next to classical baselines, all fitted and compared the
same way, with circuits for each.

- [Tutorial: question-order effects](../tutorials/02_question_order.ipynb)
- [Tutorial: trust on a qubit](../tutorials/03_trust_on_a_qubit.ipynb)
- [Model atlas: quantum-like families](../atlas/quantum_like.md)

## References

- Busemeyer, J. R., & Bruza, P. D. (2024). *Quantum Models of Cognition and Decision* (2nd ed.).
  Cambridge University Press.
- Pothos, E. M., & Busemeyer, J. R. (2013). Can quantum probability provide a new direction for
  cognitive modeling? *Behavioral and Brain Sciences*, 36(3), 255-274.
- Wang, Z., & Busemeyer, J. R. (2013). A quantum question order model supported by empirical tests of
  an a priori and precise prediction. *Topics in Cognitive Science*, 5(4), 689-710.
- Wang, Z., Solloway, T., Shiffrin, R. M., & Busemeyer, J. R. (2014). Context effects produced by
  question orders reveal quantum nature of human judgments. *PNAS*, 111(26), 9431-9436.
- Pothos, E. M., & Busemeyer, J. R. (2022). Quantum cognition. *Annual Review of Psychology*, 73,
  749-778.
