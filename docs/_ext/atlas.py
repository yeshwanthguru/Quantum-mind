"""Model atlas: one block diagram per model, generated into docs/atlas/ (python docs/_ext/atlas.py).

Each entry gives the model's purpose, a Mermaid block diagram (inputs, internal steps, outputs and the
classical baselines it is compared with), the API object and a reference. Node classes:
``in`` inputs and data, ``op`` internal steps, ``out`` outputs, ``hum`` people, ``base`` baselines.
"""
import importlib
import pathlib

DOCS = pathlib.Path(__file__).resolve().parents[1]
OUT = DOCS / 'atlas'

STYLE = """    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef hum fill:#3d1414,stroke:#ff7b72,color:#e6edf3
    classDef base fill:#262626,stroke:#8b949e,color:#e6edf3,stroke-dasharray:4 3"""


def M(title, api, summary, diagram, baselines=None, ref=None, tutorial=None):
    """One atlas entry."""
    return dict(title=title, api=api, summary=summary, diagram=diagram, baselines=baselines, ref=ref,
                tutorial=tutorial)


PAGES = {}

# --------------------------------------------------------------------------------------------- core
PAGES['core'] = ('Core: states, fitting and comparison',
                 'The building blocks every model uses: linear algebra for states and projectors, the '
                 'model base class, maximum-likelihood fitting, model comparison and recovery studies.', [
    M('Lüders rule and answer sequences', ['quantum_mind.core.linalg.luders', 'quantum_mind.core.linalg.sequence_probabilities'],
      'The probability of a sequence of yes/no answers when each answer projects the belief.',
      '''flowchart LR
    psi["belief ψ"]:::hum --> P1["project on answer 1<br/>P₁ψ"]:::op --> N1["renormalise"]:::op --> P2["project on answer 2"]:::op --> pr["‖P₂P₁ψ‖²<br/>sequence probability"]:::out''',
      None, 'Lüders, G. (1951). Annalen der Physik, 443, 322-328.'),
    M('Open-system evolution', ['quantum_mind.core.linalg.lindblad_superoperator', 'quantum_mind.core.linalg.evolve_density'],
      'Density matrices that rotate and lose coherence over time (Lindblad equation).',
      '''flowchart LR
    rho["density matrix ρ"]:::in --> H["Hamiltonian H<br/>coherent rotation"]:::op
    rho --> L["Lindblad operators<br/>dephasing γ"]:::op
    H & L --> E["exp(t·𝓛) ρ"]:::op --> out["ρ(t)"]:::out''',
      None, 'Breuer, H.-P., & Petruccione, F. (2002). The Theory of Open Quantum Systems. Oxford University Press.'),
    M('Model and parameters', ['quantum_mind.core.model.Model', 'quantum_mind.core.model.Param'],
      'Base class: declares parameters with ranges, predicts answer distributions, simulates data and '
      'computes the log-likelihood.',
      '''flowchart LR
    th["free parameters<br/>(unbounded vector)"]:::in --> map["Param transforms<br/>prob · angle · bounded"]:::op --> m["model.predict(design)"]:::op
    m --> ll["loglik(data)"]:::out
    m --> s["sample(design, n)"]:::out''',
      None, None),
    M('Fit, compare and recover', ['quantum_mind.core.fit.fit', 'quantum_mind.core.fit.compare', 'quantum_mind.core.fit.recovery'],
      'Multi-start maximum likelihood; AIC/BIC comparison; parameter and model recovery studies.',
      '''flowchart LR
    D[("counts per condition")]:::in --> F["multi-start optimiser<br/>Nelder-Mead"]:::op --> R["FitResult<br/>params · loglik · AIC · BIC"]:::out
    R --> C{"compare<br/>lowest BIC wins"}:::op
    G["simulate from each model"]:::in --> F
    C --> REC["recovery matrix"]:::out''',
      'every model is compared with its classical baselines this way',
      'Burnham, K. P., & Anderson, D. R. (2002). Model Selection and Multimodel Inference (2nd ed.). Springer.',
      '01_introduction'),
    M('Individual differences', ['quantum_mind.core.fit.fit_individuals', 'quantum_mind.core.fit.compare_individuals'],
      'Fit each person separately and compare pooled with per-person models.',
      '''flowchart LR
    P[("data per person")]:::hum --> I["fit each person"]:::op --> S["sum of BICs"]:::out
    P --> Po["pooled fit"]:::base --> S2["pooled BIC"]:::out
    S & S2 --> V{"which explains people better?"}:::op''',
      'pooled model', None),
    M('Uncertainty, online updating and informative questions',
      ['quantum_mind.core.uncertainty.bootstrap', 'quantum_mind.core.online.OnlinePersonModel',
       'quantum_mind.core.design.information_gain', 'quantum_mind.core.design.rank_conditions',
       'quantum_mind.core.design.model_posterior'],
      'Three tools that work with every model: bootstrap intervals for parameters and predictions, a '
      'per-person posterior updated after every answer, and the choice of the condition (question, '
      'order or display) that best tells competing models apart.',
      '''flowchart LR
    F["fitted model<br/>(population)"]:::in --> B["bootstrap<br/>simulate · refit"]:::op --> CI["parameter and<br/>predictive intervals"]:::out
    F --> O["particles around the estimate"]:::op
    A["one answer from this person"]:::hum --> O --> PP["per-person posterior<br/>predictive"]:::out
    PP & M2["competing models"]:::base --> IG["information gain per condition<br/>H(mix) − Σ w H(p)"]:::op --> Q["ask the most<br/>informative question"]:::out
    A --> MP["model posterior"]:::out''',
      'point estimates of one pooled fit (how the source papers report the models)',
      'Myung, J. I., & Pitt, M. A. (2009). Psychological Review, 116(3), 499-518; Chopin, N. (2002). '
      'Biometrika, 89(3), 539-551; Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.'),
])

# ------------------------------------------------------------------------------------ quantum-like
PAGES['quantum_like'] = ('Quantum-like families',
                         'Models of human judgement, decision, memory and trust written in quantum probability, '
                         'each with the classical baselines it must beat.', [
    M('Question order: quantum-like model', ['quantum_mind.families.order_effects.models.QuantumOrderModel',
                                             'quantum_mind.families.order_effects.models.QuantumOrderModel4D'],
      'Belief and "yes" subspaces in a 3D or 4D real space; answering projects the belief, so order matters.',
      '''flowchart LR
    psi["belief ψ (angle g)"]:::hum --> A["subspace A (angle a)"]:::op & B["subspace B (angle b)"]:::op
    A --> AB["A then B<br/>‖P_B P_A ψ‖²"]:::op
    B --> BA["B then A<br/>‖P_A P_B ψ‖²"]:::op
    AB & BA --> out["joint answers in both orders"]:::out
    out --> qq["QQ test: same-answer rate equal"]:::out
    base1["Bayes · anchoring · saturated"]:::base -.-> out''',
      'BayesOrderModel (no order effect), AnchoringOrderModel, SaturatedOrderModel',
      'Wang, Z., & Busemeyer, J. R. (2013). Topics in Cognitive Science, 5(4), 689-710.', '02_question_order'),
    M('Question order: classical baselines', ['quantum_mind.families.order_effects.models.BayesOrderModel',
                                              'quantum_mind.families.order_effects.models.AnchoringOrderModel',
                                              'quantum_mind.families.order_effects.models.SaturatedOrderModel'],
      'Bayes: a fixed joint distribution; anchoring: the second answer is pulled towards the first; '
      'saturated: one free probability per cell (upper bound on fit).',
      '''flowchart LR
    pA["P(A) · P(B) · correlation"]:::in --> J["fixed joint table"]:::base --> o1["same in both orders"]:::out
    w["anchoring weights wA, wB"]:::in --> AN["second answer pulled<br/>towards the first"]:::base --> o2["order effect"]:::out''',
      None, 'Hogarth, R. M., & Einhorn, H. J. (1992). Cognitive Psychology, 24, 1-55.', '02_question_order'),
    M('Conjunction fallacy', ['quantum_mind.families.conjunction.models.QuantumConjunctionModel'],
      '"Linda is a feminist bank teller" judged likelier than "bank teller": sequential projection '
      'onto the likely event first can exceed the single probability.',
      '''flowchart LR
    psi["belief about Linda"]:::hum --> F["project on 'feminist'<br/>(likely)"]:::op --> BT["then on 'bank teller'"]:::op --> pc["P(F and BT)"]:::out
    psi --> BT2["project on 'bank teller'"]:::op --> pb["P(BT)"]:::out
    pc & pb --> cmp{"P(F∧BT) > P(BT)?"}:::out
    base["classical joint · averaging · probability theory plus noise"]:::base -.-> cmp''',
      'ClassicalJointModel, AveragingModel, PTNModel',
      'Busemeyer, J. R., Pothos, E. M., Franco, R., & Trueblood, J. S. (2011). Psychological Review, 118(2), 193-218.'),
    M('Interference and the disjunction effect', ['quantum_mind.families.interference.models.InterferenceModel'],
      'Probability of a choice when an intermediate outcome is unknown differs from the average over '
      'known outcomes: an interference term with a phase.',
      '''flowchart LR
    k1["outcome known: win"]:::in --> p1["P(play | win)"]:::op
    k2["outcome known: lose"]:::in --> p2["P(play | lose)"]:::op
    u["outcome unknown"]:::hum --> amp["amplitudes add<br/>+ 2√(..)cos θ"]:::op --> pu["P(play | unknown)"]:::out
    p1 & p2 --> tot["law of total probability"]:::base -.-> pu''',
      'ClassicalMixtureModel', 'Pothos, E. M., & Busemeyer, J. R. (2009). Proceedings of the Royal Society B, 276, 2171-2178.'),
    M('Quantum-like Bayesian network', ['quantum_mind.families.qlbn.models.QLBNModel'],
      'A Bayesian network whose unobserved nodes are summed as amplitudes, giving interference terms.',
      '''flowchart LR
    G["network graph<br/>CPTs"]:::in --> H["hidden configurations"]:::op --> A["sum amplitudes<br/>with phases"]:::op --> m["marginal of the query"]:::out
    G --> C["classical BN<br/>sum probabilities"]:::base -.-> m''',
      'ClassicalBNModel', 'Moreira, C., & Wichert, A. (2016). Frontiers in Psychology, 7, 11.'),
    M('Belief dynamics: quantum and Markov walks', ['quantum_mind.families.dynamics.models.QuantumWalk',
                                                   'quantum_mind.families.dynamics.models.OpenSystemWalk'],
      'Evidence accumulation as a walk over confidence levels: unitary (quantum), with decoherence '
      '(open system) or stochastic (Markov).',
      '''flowchart LR
    s0["initial confidence"]:::hum --> U["Hamiltonian walk<br/>drift · diffusion"]:::op --> D["dephasing γ"]:::op --> j["confidence at time t"]:::out
    s0 --> MK["Markov walk"]:::base -.-> j
    j --> Q["asking changes later answers"]:::out''',
      'MarkovWalk', 'Kvam, P. D., Pleskac, T. J., Yu, S., & Busemeyer, J. R. (2015). PNAS, 112(34), 10645-10650.'),
    M('Trust belief (open system)', ['quantum_mind.families.dynamics.models.OpenSystemBelief'],
      'Trust as a qubit: outcomes rotate it, time dephases it, a trust question projects it.',
      '''flowchart LR
    t0["trust state ρ₀"]:::hum --> ev{"hand-over outcome"}:::in
    ev -->|good| up["rotate by a₊"]:::op
    ev -->|bad| dn["rotate by −a₋"]:::op
    up & dn --> deph["dephase γ"]:::op --> q{"asked?"}:::op
    q -->|yes| proj["project · answer"]:::out
    q -->|no| nxt["next interaction"]:::op
    mk["Markov belief"]:::base -.-> proj''',
      'MarkovBelief', 'Busemeyer, J. R., & Bruza, P. D. (2024). Quantum Models of Cognition and Decision (2nd ed.); '
      'Roeder, L., et al. (2023). A quantum model of trust calibration in human-AI interactions. Entropy, 25(9), 1362.',
      '03_trust_on_a_qubit'),
    M('Quantum decision theory', ['quantum_mind.families.decision.models.QDTModel'],
      'Choice probability = utility factor + attraction (interference) factor.',
      '''flowchart LR
    L["lotteries"]:::in --> U["utility factor f"]:::op
    L --> AT["attraction factor q<br/>uncertainty aversion"]:::op
    U & AT --> P["P(choice) = f + q"]:::out
    EU["expected utility · prospect theory"]:::base -.-> P''',
      'ExpectedUtilityModel, ProspectTheoryModel', 'Yukalov, V. I., & Sornette, D. (2011). Theory and Decision, 70, 283-328.'),
    M('Contextuality tests', ['quantum_mind.families.contextuality.models.chsh',
                              'quantum_mind.families.contextuality.models.cyclic_contextuality'],
      'Is there one joint distribution behind all the measurements? CHSH and Contextuality-by-Default.',
      '''flowchart LR
    d[("paired measurements<br/>in 4 contexts")]:::in --> c["correlations"]:::op --> s["S statistic"]:::op
    s --> b{"S > 2 (CHSH)<br/>or CbD criterion"}:::out
    cl["classical bound"]:::base -.-> b''',
      'classical (noncontextual) bounds', 'Dzhafarov, E. N., Kujala, J. V., & Cervantes, V. H. (2016). Lecture Notes in Computer Science, 9535, 12-23.'),
    M('Asymmetric similarity', ['quantum_mind.families.similarity.models.QuantumSimilarityModel'],
      'Similarity of A to B as projection between concept subspaces of different dimension; '
      '"Korea is like China" ≠ "China is like Korea".',
      '''flowchart LR
    A["concept A subspace<br/>(small)"]:::in --> PA["project A onto B"]:::op --> sAB["sim(A, B)"]:::out
    B["concept B subspace<br/>(large)"]:::in --> PB["project B onto A"]:::op --> sBA["sim(B, A)"]:::out
    g["geometric · biased geometric"]:::base -.-> sAB''',
      'GeometricModel, BiasedGeometricModel', 'Pothos, E. M., Busemeyer, J. R., & Trueblood, J. S. (2013). Psychological Review, 120(3), 679-696.'),
    M('Quantum games', ['quantum_mind.families.game_theory.models.EWLGame'],
      'Eisert-Wilkens-Lewenstein games: players apply strategies to an entangled pair.',
      '''flowchart LR
    s["|00⟩"]:::in --> J["entangler J"]:::op --> UA["player A: U(θ, φ)"]:::hum & UB["player B: U(θ, φ)"]:::hum
    UA & UB --> Jd["J†"]:::op --> M2["measure → payoffs"]:::out
    N["classical Nash equilibria"]:::base -.-> M2''',
      'classical_nash_equilibria', 'Eisert, J., Wilkens, M., & Lewenstein, M. (1999). Physical Review Letters, 83, 3077.'),
    M('Episodic memory overdistribution', ['quantum_mind.families.memory.models.QuantumEpisodicModel'],
      'Verbatim and gist traces as non-orthogonal subspaces; recall of "target or related" can exceed '
      'the sum of its parts.',
      '''flowchart LR
    cue["memory probe"]:::hum --> V["verbatim subspace"]:::op & G["gist subspace"]:::op
    V & G --> P["projection onto V ⊕ G"]:::op --> o["P(remember) · overdistribution"]:::out
    add["additive verbatim + gist"]:::base -.-> o''',
      'AdditiveMemoryModel', 'Brainerd, C. J., Wang, Z., & Reyna, V. F. (2013). Psychological Review, 120(1), 1-33.'),
    M('Concept combination (Fock space)', ['quantum_mind.families.concepts.models.FockSpaceConceptModel'],
      '"Pet fish" membership: a superposition of the two concepts plus their combination sector.',
      '''flowchart LR
    A["membership in A"]:::in --> F1["sector 1: (A + B)/√2<br/>with interference"]:::op
    B["membership in B"]:::in --> F1
    A & B --> F2["sector 2: A and B"]:::op
    F1 & F2 --> mu["μ(A and B)<br/>overextension"]:::out
    cl["product · minimum · weighted average"]:::base -.-> mu''',
      'ProductConceptModel, MinConceptModel, WeightedAverageModel', 'Aerts, D. (2009). Journal of Mathematical Psychology, 53(5), 314-348.'),
    M('Bistable perception (quantum Zeno)', ['quantum_mind.families.perception.models.QuantumZenoBistableModel'],
      'A percept rotating between two interpretations, re-projected every Δt; dwell time ∝ Δt/sin²(gΔt).',
      '''flowchart LR
    p0["percept: interpretation 1"]:::hum --> R["rotate by gΔt"]:::op --> C{"check every Δt"}:::op
    C -->|stays| R
    C -->|flips| f["dwell time ends"]:::out
    f --> d["dwell-time distribution"]:::out
    mk["Markov switching · gamma renewal"]:::base -.-> d''',
      'MarkovSwitchingModel, GammaRenewalModel', 'Atmanspacher, H., Filk, T., & Römer, H. (2004). Biological Cybernetics, 90, 33-40.',
      '09_bistable_perception'),
])

# ------------------------------------------------------------------------------------------ robotics
PAGES['robotics'] = ('Robot decision layer',
                     'Components that turn models of people and uncertain perception into decisions: ask or '
                     'act, calibrate, route, plan questions, personalise and hand over.', [
    M('Human-model ensemble', ['quantum_mind.applications.robotics.HumanModelEnsemble'],
      'Competing models of how people answer, weighted by evidence; reports the predicted answers and '
      'how much the models disagree.',
      '''flowchart LR
    D[("answers so far")]:::hum --> F["fit each model<br/>QL · Bayes · anchoring"]:::op --> W["BIC weights"]:::op
    W --> P["weighted prediction"]:::out
    W --> U["disagreement (bits)"]:::out''',
      'single-model prediction', 'Wagenmakers, E.-J., & Farrell, S. (2004). Psychonomic Bulletin & Review, 11, 192-196.',
      '04_ask_or_act'),
    M('Ask or act', ['quantum_mind.applications.robotics.ask_or_act', 'quantum_mind.applications.robotics.risk_aware_ask_or_act'],
      'Ask when the expected cost of acting (error cost × (1 − p), by hazard) exceeds the cost of asking, '
      'or when the human models disagree.',
      '''flowchart LR
    p["confidence p<br/>(or lower bound)"]:::in --> E["error cost × (1 − p)"]:::op
    hz["hazard level"]:::in --> E
    d["model disagreement"]:::in --> G{"disagree?"}:::op
    E --> C{"vs ask cost"}:::op
    G -->|yes| ask["ask"]:::out
    C -->|higher| ask
    C -->|lower| act["act"]:::out''',
      None, 'Howard, R. A. (1966). IEEE Transactions on Systems Science and Cybernetics, 2(1), 22-26.', '04_ask_or_act'),
    M('Human-model service', ['quantum_mind.applications.robotics.HumanModelService'],
      'Middleware-independent service around the ensemble, used by the ROS 2 node.',
      '''flowchart LR
    ros["ROS 2 node / any caller"]:::in --> S["HumanModelService"]:::op --> E["ensemble update · predict"]:::op --> R["prediction + decision"]:::out''',
      None, None),
    M('Intent resolution', ['quantum_mind.applications.intent.QuantumIntentResolver',
                            'quantum_mind.applications.intent.BayesIntentResolver'],
      'Which object does the person mean? Cues as bases in a shared space; incompatible cues give '
      'order-dependent posteriors.',
      '''flowchart LR
    c1["cue 1 (speech)"]:::in --> R1["rotate to cue basis<br/>angle θ₁"]:::op
    c2["cue 2 (gaze)"]:::in --> R2["angle θ₂"]:::op
    R1 & R2 --> PO["posterior over intents"]:::out
    BY["Bayesian product of likelihoods"]:::base -.-> PO''',
      'BayesIntentResolver', 'Busemeyer & Bruza (2024), ch. 4.', '08_fusion'),
    M('Recalibration', ['quantum_mind.applications.calibration.PlattScaling',
                        'quantum_mind.applications.calibration.TemperatureScaling',
                        'quantum_mind.applications.calibration.IsotonicCalibration'],
      'Map raw scores to calibrated probabilities, fitted on held-out data.',
      '''flowchart LR
    s["raw scores / logits"]:::in --> m{"method"}:::op
    m --> pl["Platt: sigmoid(a·s + b)"]:::op
    m --> ts["temperature: softmax(z / T)"]:::op
    m --> iso["isotonic: monotone fit (PAV)"]:::op
    pl & ts & iso --> p["calibrated probability"]:::out
    p --> ece["ECE · Brier · reliability diagram"]:::out''',
      'raw scores', 'Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). Proceedings of ICML.', '05_calibration'),
    M('Conformal prediction sets', ['quantum_mind.applications.calibration.SplitConformalClassifier',
                                    'quantum_mind.applications.calibration.AdaptiveConformalSets'],
      'Sets of labels that contain the truth with probability ≥ 1 − α for exchangeable data; the adaptive '
      'version keeps its long-run coverage for one person\'s answers while that person changes.',
      '''flowchart LR
    cal[("calibration data")]:::in --> sc["non-conformity scores"]:::op --> q["(1 − α) quantile"]:::op
    new["new scores"]:::in --> S["labels with score ≤ q"]:::op
    q --> S --> set["prediction set<br/>size > 1 → ask"]:::out''',
      'top-1 prediction', 'Angelopoulos, A. N., & Bates, S. (2023). Foundations and Trends in Machine Learning, 16(4), 494-591; '
      'Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift. NeurIPS 34.',
      '05_calibration'),
    M('Calibration monitor', ['quantum_mind.applications.calibration.CalibrationMonitor',
                              'quantum_mind.applications.calibration.clopper_pearson'],
      'Running accuracy per confidence bin with an exact lower bound.',
      '''flowchart LR
    o["(confidence, correct) stream"]:::in --> b["bin counts"]:::op --> cp["Clopper-Pearson bound"]:::op --> lb["p_lower for ask-or-act"]:::out''',
      None, 'Clopper, C. J., & Pearson, E. S. (1934). Biometrika, 26, 404-413.', '05_calibration'),
    M('Confidence gate', ['quantum_mind.applications.orchestration.ConfidenceGate'],
      'Pick the module with the best calibrated confidence minus λ × cost, within a budget.',
      '''flowchart LR
    m1["module 1 conf, cost"]:::in & m2["module 2 conf, cost"]:::in & m3["module 3 conf, cost"]:::in --> cal["calibrate"]:::op --> sc["conf − λ·cost"]:::op
    bud["budget window"]:::in --> sc
    sc --> pick["selected module"]:::out''',
      'highest raw confidence', None, '06_orchestration'),
    M('Meta-calibrated gate', ['quantum_mind.applications.orchestration.MetaCalibratedGate',
                               'quantum_mind.applications.orchestration.meta_fit_offsets'],
      'Learn each module\'s typical confidence bias from earlier tasks and use it as a prior on a new task.',
      '''flowchart LR
    logs[("logs of earlier tasks")]:::in --> eb["empirical Bayes:<br/>mean bias, prior strength"]:::op
    eb --> g["gate on new task"]:::op
    fb["successes on new task"]:::in --> up["posterior bias update"]:::op --> g
    g --> sel["module choice"]:::out
    sc["learn from scratch"]:::base -.-> sel''',
      'gate without a prior', 'Efron, B., & Morris, C. (1975). JASA, 70, 311-319.', '06_orchestration'),
    M('Question planner', ['quantum_mind.applications.questioning.QuestionPlanner'],
      'Choose the question with the highest value of information, or act when no question is worth asking.',
      '''flowchart LR
    pr["posterior over hypotheses"]:::in --> v["value of information<br/>per question"]:::op --> d{"best VOI > 0?"}:::op
    am["answer model<br/>projective or independent"]:::in --> v
    d -->|yes| ask["ask it"]:::hum --> up["update posterior"]:::op --> pr
    d -->|no| act["act on the MAP hypothesis"]:::out''',
      'IndependentAnswerModel (order-free)', 'Howard (1966); Wang & Busemeyer (2013); related: Rosenthal, S., Dey, A. K., & Veloso, M. (2009). How robots\' '
      'questions affect the accuracy of the human responses. IEEE RO-MAN.', '07_question_planner'),
    M('Answer models', ['quantum_mind.applications.questioning.ProjectiveAnswerModel',
                        'quantum_mind.applications.questioning.IndependentAnswerModel'],
      'How a person with a given intent answers a sequence of questions.',
      '''flowchart LR
    h["intent state"]:::hum --> pq["projectors per question"]:::op --> L["sequence likelihood<br/>(order-dependent)"]:::out
    h --> ind["independent per-question P(yes)"]:::base --> L2["order-free likelihood"]:::out''',
      None, None, '07_question_planner'),
    M('Personalised human models', ['quantum_mind.applications.personalisation.PopulationPrior',
                                    'quantum_mind.applications.personalisation.PersonalisedHumanModel',
                                    'quantum_mind.applications.personalisation.fit_map'],
      'Partial pooling: each person\'s parameters are shrunk towards the population, most when data are few.',
      '''flowchart LR
    pop[("many people")]:::hum --> pp["population prior<br/>mean, covariance"]:::op
    me[("this person's answers")]:::hum --> map["MAP fit<br/>or online posterior"]:::op
    pp --> map --> pers["personal model<br/>with credible intervals"]:::out
    ind["individual fit"]:::base -.-> pers
    one["one model for all"]:::base -.-> pers''',
      'population model, individual fits', 'Gelman, A., & Hill, J. (2007). Data Analysis Using Regression and Multilevel/Hierarchical Models.'),
    M('Trust-aware hand-over', ['quantum_mind.applications.handover.TrustAwareHandover'],
      'Before each hand-over, choose hand over, slow hand-over, ask or wait by expected cost under the '
      'trust model.',
      '''flowchart LR
    b["trust belief"]:::hum --> c["expected cost of<br/>hand over · slow · ask · wait"]:::op --> a["chosen action"]:::out
    a --> o["outcome / answer"]:::in --> u["update belief"]:::op --> b
    mk["Markov-based policy · always hand over"]:::base -.-> a''',
      'Markov-based policy, always hand over', 'Hancock, P. A., et al. (2011). Human Factors, 53(5), 517-527.',
      '03_trust_on_a_qubit'),
])

# ---------------------------------------------------------------------------------------- perception
PAGES['perception'] = ('Perception and fusion',
                       'Adapters that turn perception outputs into likelihoods, three fusion rules, and the '
                       'bistable-perception models.', [
    M('Perception adapters', ['quantum_mind.applications.fusion.detector_likelihood',
                              'quantum_mind.applications.fusion.asr_likelihood',
                              'quantum_mind.applications.fusion.direction_likelihood'],
      'Detector scores or logits, speech n-best lists and gaze or pointing directions as likelihoods over '
      'the same candidate objects.',
      '''flowchart LR
    det["detector scores / logits"]:::in --> a1["tempered softmax"]:::op
    asr["speech n-best"]:::in --> a2["keyword match × confidence"]:::op
    dir["gaze / pointing vector"]:::in --> a3["von Mises-Fisher"]:::op
    a1 & a2 & a3 --> L["likelihoods over objects"]:::out''',
      None, 'Fisher, R. A. (1953). Proceedings of the Royal Society A, 217, 295-305.', '08_fusion'),
    M('Fusion rules', ['quantum_mind.applications.fusion.bayes_fusion',
                       'quantum_mind.applications.fusion.dempster_shafer_fusion',
                       'quantum_mind.applications.fusion.compare_fusion'],
      'Combine cues three ways and compare them on held-out data.',
      '''flowchart LR
    L["cue likelihoods"]:::in --> B["Bayes: product"]:::base
    L --> DS["Dempster-Shafer:<br/>masses + ignorance"]:::base
    L --> Q["quantum-like:<br/>cue bases, order"]:::op
    B & DS & Q --> cmp["held-out log loss"]:::out''',
      'Bayes, Dempster-Shafer', 'Shafer, G. (1976). A Mathematical Theory of Evidence. Princeton University Press.', '08_fusion'),
    M('Cue incompatibility', ['quantum_mind.applications.fusion.fit_incompatibility'],
      'Fit the angles between cue bases from trials presented in different cue orders.',
      '''flowchart LR
    t[("trials: cue order → choice")]:::hum --> f["maximise likelihood<br/>over angles"]:::op --> th["angle per cue"]:::out''',
      None, None, '08_fusion'),
    M('Dwell-time baselines', ['quantum_mind.families.perception.models.MarkovSwitchingModel',
                               'quantum_mind.families.perception.models.GammaRenewalModel'],
      'Constant switching rate (exponential dwell times) and the gamma renewal model used in the '
      'perception literature.',
      '''flowchart LR
    r["rate λ"]:::in --> ex["exponential dwell times"]:::base --> d["dwell-time histogram"]:::out
    ks["shape k, scale θ"]:::in --> ga["gamma dwell times"]:::base --> d''',
      None, 'Brascamp, J. W., et al. (2006). Journal of Vision, 6, 1244-1256.', '09_bistable_perception'),
])

# ------------------------------------------------------------------------------------------ learning
PAGES['learning'] = ('Learning',
                     'Reinforcement-learning environments with simulated people, exploration strategies, a '
                     'quantum policy and network compression.', [
    M('Clarification environment', ['quantum_mind.envs.hri.ClarificationEnv'],
      'Resolve an ambiguous request by asking yes/no questions (order-dependent answers), then act.',
      '''flowchart LR
    ag["agent"]:::op -->|ask q| P["simulated person<br/>projective answers"]:::hum -->|±1| obs["answers so far"]:::in --> ag
    ag -->|act on object| R["reward: +1 right · −5 wrong<br/>−0.3 per question"]:::out''',
      None, 'Towers, M., et al. (2024). Gymnasium. arXiv:2407.17032.', '10_rl_environments'),
    M('Trust hand-over environment', ['quantum_mind.envs.hri.TrustHandoverEnv', 'quantum_mind.envs.hri.sample_people'],
      'Hand objects to a person whose trust follows the trust qubit; a new simulated person can be drawn '
      'for every episode from the uncertainty of a fitted model.',
      '''flowchart LR
    ag["agent"]:::op -->|hand over · slow · ask · wait| P["person: trust qubit"]:::hum
    P --> o["outcome · answer"]:::in --> ag
    P --> R["reward: − failure, slow, ask, wait costs"]:::out''',
      None, None, '10_rl_environments'),
    M('Grid world', ['quantum_mind.envs.hri.GridWorldEnv', 'quantum_mind.inspired.rl.GridWorld'],
      'Navigation to a goal, the standard test bed for tabular agents.',
      '''flowchart LR
    s["cell"]:::in --> a["move"]:::op --> s2["next cell"]:::in --> g{"goal?"}:::out''',
      None, 'Sutton & Barto (2018).', '10_rl_environments'),
    M('Exploration strategies', ['quantum_mind.inspired.exploration.AmplitudeExploration',
                                 'quantum_mind.inspired.exploration.EpsilonGreedy',
                                 'quantum_mind.inspired.exploration.Boltzmann', 'quantum_mind.inspired.exploration.UCB'],
      'Amplitude exploration samples actions with Born-rule probabilities and rotates amplitudes by the '
      'advantage; compared with ε-greedy, softmax and UCB.',
      '''flowchart LR
    psi["amplitudes ψ(s, ·)"]:::op --> born["sample a with |ψₐ|²"]:::op --> env["environment"]:::in --> td["TD error / advantage"]:::op
    td --> rot["rotate ψₐ, keep a floor"]:::op --> psi
    eg["ε-greedy · softmax · UCB"]:::base -.-> born''',
      'EpsilonGreedy, Boltzmann, UCB', 'Dong, D., et al. (2008). IEEE TSMC-B, 38(5), 1207-1220.', '10_rl_environments'),
    M('Tabular agent', ['quantum_mind.inspired.exploration.TabularAgent', 'quantum_mind.inspired.exploration.run_episodes'],
      'Q-learning with a pluggable explorer.',
      '''flowchart LR
    s["state"]:::in --> X["explorer picks a"]:::op --> E["env step"]:::in --> Qu["Q ← Q + α·TD"]:::op --> s
    Qu --> X''',
      None, 'Watkins, C. J. C. H., & Dayan, P. (1992). Machine Learning, 8, 279-292.', '10_rl_environments'),
    M('Quantum-inspired Q-learning', ['quantum_mind.inspired.rl.QuantumInspiredQLearning'],
      'Action selection by amplitude collapse with Grover-style amplitude amplification.',
      '''flowchart LR
    s["state"]:::in --> amp["action amplitudes"]:::op --> col["collapse → action"]:::op --> r["reward"]:::in --> gr["amplify good action"]:::op --> amp
    ql["Q-learning"]:::base -.-> col''',
      'QLearning', 'Dong, D., Chen, C., Li, H., & Tarn, T.-J. (2008). IEEE TSMC-B, 38(5), 1207-1220.'),
    M('Variational quantum policy', ['quantum_mind.quantum.policy.VariationalPolicy', 'quantum_mind.quantum.policy.reinforce'],
      'Action probabilities from a data re-uploading circuit, trained with REINFORCE and adjoint gradients.',
      '''flowchart LR
    s["state features"]:::in --> enc["RY(x) encoding"]:::op --> var["trainable RY · RZ · CZ<br/>(re-uploaded)"]:::op --> meas["Born probabilities<br/>of read-out qubits"]:::op --> a["action"]:::out
    a --> G["returns − baseline"]:::in --> grad["adjoint gradient"]:::op --> var
    lin["linear softmax policy"]:::base -.-> a''',
      'linear softmax policy (tutorial 11)', 'Jerbi, S., et al. (2021). NeurIPS 34.', '11_quantum_policy'),
    M('Tensor-train layers', ['quantum_mind.inspired.tensor_layers.TTMatrix', 'quantum_mind.inspired.tensor_layers.compress_layers'],
      'A weight matrix stored as a chain of small cores (TT-SVD); applied without forming the dense matrix.',
      '''flowchart LR
    W["dense W (M × N)"]:::in --> rs["reshape to (m₁n₁)(m₂n₂)…"]:::op --> svd["truncated SVDs<br/>max rank r"]:::op --> cores["cores G₁ … G_d"]:::out
    x["inputs"]:::in --> mv["contract core by core"]:::op --> y["W x"]:::out
    cores --> mv
    dense["dense layer"]:::base -.-> y''',
      'dense layer', 'Oseledets, I. V. (2011). SIAM J. Sci. Comput., 33(5), 2295-2317.', '12_tensor_train'),
    M('Quanvolutional filter', ['quantum_mind.quantum.quanvolution.QuanvolutionFilter'],
      'Each 2 × 2 image patch is encoded into four qubits, passed through a fixed random circuit and read '
      'out as four ⟨Z⟩ channels.',
      '''flowchart LR
    img["image"]:::in --> pt["2×2 patches"]:::op --> enc["RY(πx) per pixel"]:::op --> rc["random circuit"]:::op --> z["⟨Z⟩ per qubit"]:::op --> fm["4-channel feature map"]:::out --> clf["classical classifier"]:::out
    rnd["random classical filter · raw pixels"]:::base -.-> clf''',
      'RandomConvFilter, raw pixels', 'Henderson, M., et al. (2020). Quantum Machine Intelligence, 2, 2.', '13_quanvolution'),
])

# ------------------------------------------------------------------------------------------- quantum
PAGES['quantum'] = ('Quantum models',
                    'Gate-model machine learning and algorithms on the built-in simulator, exportable to '
                    'Qiskit for Aer, IBM Quantum and Amazon Braket.', [
    M('State-vector simulator', ['quantum_mind.quantum.statevector.Circuit'],
      'Batched NumPy simulation of parameterised circuits with adjoint gradients and Qiskit export.',
      '''flowchart LR
    c["gates with angles<br/>W(k) · X(j) · constants"]:::in --> sim["batched state vectors"]:::op --> p["probabilities · ⟨Z⟩"]:::out
    sim --> g["adjoint gradients"]:::out
    c --> qk["to_qiskit()"]:::out''',
      None, 'Jones, T., & Gacon, J. (2020). arXiv:2009.02823 (adjoint differentiation).', '01_introduction'),
    M('Feature maps and ansatz', ['quantum_mind.quantum.ansatz.angle_encoding', 'quantum_mind.quantum.ansatz.zz_feature_map',
                                  'quantum_mind.quantum.ansatz.hardware_efficient', 'quantum_mind.quantum.ansatz.parameter_shift'],
      'How data enter a circuit and which trainable layers follow.',
      '''flowchart LR
    x["features"]:::in --> ang["angle encoding RY(x)"]:::op
    x --> zz["ZZ feature map<br/>H · RZ · RZZ"]:::op
    ang & zz --> he["hardware-efficient layers<br/>RY · RZ · ring of CZ"]:::op --> o["state"]:::out
    o --> ps["parameter-shift gradient"]:::out''',
      None, 'Havlíček, V., et al. (2019). Nature, 567, 209-212.'),
    M('Variational classifier and regressor', ['quantum_mind.quantum.classifiers.VariationalClassifier',
                                               'quantum_mind.quantum.classifiers.VariationalRegressor'],
      'Data re-uploading circuit trained by gradient descent on cross-entropy or squared error.',
      '''flowchart LR
    x["features"]:::in --> c["re-uploading circuit"]:::op --> p["class probabilities<br/>or ⟨Z⟩"]:::out
    p --> L["loss"]:::op --> gd["adjoint gradient step"]:::op --> c
    lr["logistic / ridge regression"]:::base -.-> p''',
      'logistic regression, ridge regression', 'Pérez-Salinas, A., et al. (2020). Quantum, 4, 226.'),
    M('Quantum kernel methods', ['quantum_mind.quantum.classifiers.QuantumKernel',
                                 'quantum_mind.quantum.classifiers.QuantumKernelClassifier',
                                 'quantum_mind.quantum.classifiers.QuantumKernelAnomalyDetector',
                                 'quantum_mind.quantum.classifiers.QuantumKernelClustering'],
      'Similarity as the overlap of encoded states, used for classification, anomaly detection and clustering.',
      '''flowchart LR
    x1["x"]:::in --> e1["|φ(x)⟩"]:::op
    x2["x'"]:::in --> e2["|φ(x')⟩"]:::op
    e1 & e2 --> K["K = |⟨φ(x)|φ(x')⟩|²"]:::op --> svm["SVM · one-class · spectral"]:::out
    rbf["RBF kernel"]:::base -.-> svm''',
      'rbf_kernel', 'Schuld, M., & Killoran, N. (2019). Physical Review Letters, 122, 040504.'),
    M('QAOA', ['quantum_mind.quantum.algorithms.QAOA'],
      'Alternating cost and mixer layers for QUBO problems, with classically tuned angles.',
      '''flowchart LR
    q["QUBO"]:::in --> H["|+⟩ⁿ"]:::op --> C["cost layer e^(−iγC)"]:::op --> X["mixer e^(−iβΣX)"]:::op --> r{"repeat p times"}:::op
    r --> m["sample bit strings"]:::out
    m --> opt["tune γ, β"]:::op --> C
    sa["SA · SQA · QIEA · exact"]:::base -.-> m''',
      'simulated annealing, SQA, QIEA, brute force', 'Farhi, E., Goldstone, J., & Gutmann, S. (2014). arXiv:1411.4028.',
      '14_task_allocation'),
    M('VQE', ['quantum_mind.quantum.algorithms.VQE', 'quantum_mind.quantum.algorithms.Hamiltonian'],
      'Minimise the energy of a Pauli Hamiltonian over a parameterised state.',
      '''flowchart LR
    H["Pauli Hamiltonian"]:::in --> E["⟨ψ(θ)|H|ψ(θ)⟩"]:::op
    ans["ansatz ψ(θ)"]:::op --> E --> o["optimiser"]:::op --> ans
    E --> g["ground-state estimate"]:::out
    ex["exact diagonalisation"]:::base -.-> g''',
      'exact diagonalisation', 'Peruzzo, A., et al. (2014). Nature Communications, 5, 4213.'),
    M('Grover search', ['quantum_mind.quantum.algorithms.grover'],
      'Amplitude amplification of marked items in about √N oracle calls.',
      '''flowchart LR
    u["uniform superposition"]:::in --> O["oracle: flip marked"]:::op --> D["diffusion"]:::op --> r{"≈ π/4·√N times"}:::op --> m["measure: marked item"]:::out
    cl["classical search N/2 calls"]:::base -.-> m''',
      'classical search', 'Grover, L. K. (1996). Proceedings of STOC, 212-219.'),
    M('Quantum Fourier transform and period finding', ['quantum_mind.quantum.fourier.qft_circuit',
                                                       'quantum_mind.quantum.fourier.find_period'],
      'The QFT circuit and its use to find the period of a periodic state.',
      '''flowchart LR
    s["periodic state"]:::in --> qft["QFT: H + controlled phases"]:::op --> m["peaks at multiples of N/r"]:::op --> cf["continued fractions"]:::op --> r["period r"]:::out''',
      'classical FFT', 'Shor, P. W. (1997). SIAM Journal on Computing, 26(5), 1484-1509.'),
    M('Amplitude estimation', ['quantum_mind.quantum.estimation.AmplitudeEstimation'],
      'Estimate an expected value with fewer samples than Monte Carlo (maximum-likelihood variant).',
      '''flowchart LR
    A["state preparation A"]:::in --> G["Grover powers Qᵏ"]:::op --> M["measure for several k"]:::op --> ml["maximum likelihood"]:::op --> a["estimate of a"]:::out
    mc["Monte Carlo"]:::base -.-> a''',
      'monte_carlo_estimate', 'Suzuki, Y., et al. (2020). Quantum Information Processing, 19, 75.'),
    M('Quantum walks on networks', ['quantum_mind.quantum.walks.ctqw_probabilities',
                                    'quantum_mind.quantum.walks.quantum_walk_centrality'],
      'Continuous-time quantum walks and the centrality they induce.',
      '''flowchart LR
    G["graph adjacency"]:::in --> U["e^(−iAt)"]:::op --> p["time-averaged occupation"]:::op --> c["quantum-walk centrality"]:::out
    pr["PageRank · degree"]:::base -.-> c''',
      'pagerank, degree_centrality', 'Childs, A. M., Farhi, E., & Gutmann, S. (2002). Quantum Information Processing, 1, 35-43.'),
])

# ------------------------------------------------------------------------------------------ inspired
PAGES['inspired'] = ('Quantum-inspired algorithms',
                     'Classical algorithms that borrow quantum ideas, each next to the classical method it has '
                     'to beat.', [
    M('QIEA', ['quantum_mind.inspired.optimisers.QIEA'],
      'Q-bit individuals sampled into bit strings; amplitudes rotated towards the best solution.',
      '''flowchart LR
    Q["Q-bit angles θᵢ"]:::op --> s["sample bit strings"]:::op --> f["evaluate QUBO"]:::in --> b["best so far"]:::op --> rot["rotate θ (lookup table)"]:::op --> Q
    b --> out["solution"]:::out
    sa["simulated annealing"]:::base -.-> out''',
      'simulated_annealing', 'Han, K.-H., & Kim, J.-H. (2002). IEEE Transactions on Evolutionary Computation, 6(6), 580-593.',
      '14_task_allocation'),
    M('QPSO', ['quantum_mind.inspired.optimisers.QPSO'],
      'Particles sampled around attractors from a delta-potential-well distribution (no velocities).',
      '''flowchart LR
    pb["personal bests"]:::in & gb["global best"]:::in --> at["attractor per particle"]:::op
    mb["mean best"]:::op --> L["spread ∝ |mbest − x|"]:::op
    at & L --> x["new positions"]:::op --> f["objective"]:::in --> pb
    x --> out["minimum"]:::out''',
      'grid or random search, classical PSO', 'Sun, J., Feng, B., & Xu, W. (2004). Proceedings of IEEE CEC, 325-331.'),
    M('Simulated quantum annealing', ['quantum_mind.inspired.optimisers.SQA'],
      'Path-integral Monte Carlo on coupled replicas; a decreasing transverse field lets the search tunnel.',
      '''flowchart LR
    q["Ising problem"]:::in --> rep["P replicas"]:::op --> cp["inter-replica coupling<br/>from transverse field Γ"]:::op --> mc["Metropolis sweeps"]:::op --> an{"lower Γ"}:::op --> rep
    mc --> out["lowest-energy replica"]:::out
    sa["simulated annealing"]:::base -.-> out''',
      'simulated_annealing', 'Martoňák, R., Santoro, G. E., & Tosatti, E. (2002). Physical Review B, 66, 094203.',
      '14_task_allocation'),
    M('MPS classifier', ['quantum_mind.inspired.tensor.MPSClassifier'],
      'Features mapped to local vectors and contracted with a matrix product state to give class scores.',
      '''flowchart LR
    x["features"]:::in --> phi["local maps φ(xᵢ)"]:::op --> mps["MPS cores, bond D"]:::op --> s["class scores"]:::out
    lr["logistic regression"]:::base -.-> s''',
      'logistic regression', 'Stoudenmire, E. M., & Schwab, D. J. (2016). NeurIPS 29.'),
    M('Quantum language model', ['quantum_mind.inspired.text.QuantumLanguageModel'],
      'Documents as density matrices over term projectors; ranking by quantum relative entropy.',
      '''flowchart LR
    d["document terms"]:::in --> rho["density matrix ρ_d"]:::op
    q["query"]:::in --> rq["ρ_q"]:::op
    rho & rq --> div["divergence"]:::op --> rank["ranking"]:::out
    qlm["query likelihood (Dirichlet)"]:::base -.-> rank''',
      'QueryLikelihoodModel', 'Sordoni, A., Nie, J.-Y., & Bengio, Y. (2013). Proceedings of SIGIR, 653-662.'),
])

# ------------------------------------------------------------------------------------ infrastructure
PAGES['infrastructure'] = ('Problems, circuits and the viewer',
                           'Shared problem formats, circuits for the quantum-like families, back ends and the '
                           'Bloch-sphere viewer.', [
    M('QUBO problems', ['quantum_mind.problems.Qubo', 'quantum_mind.problems.task_allocation',
                        'quantum_mind.problems.maxcut', 'quantum_mind.problems.portfolio'],
      'One problem format for QAOA, annealing and classical heuristics.',
      '''flowchart LR
    pb["task allocation · Max-Cut · knapsack · portfolio"]:::in --> Q["QUBO: xᵀQx + offset"]:::op
    Q --> qa["QAOA"]:::out & sqa["SQA · QIEA"]:::out & sa["simulated annealing"]:::out & bf["brute force"]:::out''',
      None, 'Lucas, A. (2014). Frontiers in Physics, 2, 5.', '14_task_allocation'),
    M('Trust-aware task allocation', ['quantum_mind.applications.allocation.trust_aware_allocation',
                                      'quantum_mind.applications.allocation.expected_costs',
                                      'quantum_mind.applications.allocation.decode'],
      'Assign tasks among people working with a robot: robot-assisted tasks cost more with a person who '
      'does not trust the robot, and heavy workloads are penalised. The trust values come from the '
      'people models.',
      '''flowchart LR
    T["trust per person<br/>trust qubit · online posterior"]:::hum --> C["expected cost<br/>C + F·r·(1 − τ)"]:::op
    B[("effort per person and task")]:::in --> C
    C --> Q["QUBO + workload term"]:::op --> S["QAOA · SQA · exact"]:::op --> A["who does what"]:::out
    blind["trust-blind allocation"]:::base -.-> A''',
      'trust-blind task allocation', 'Lucas, A. (2014). Frontiers in Physics, 2, 5 (allocation QUBO); '
      'trust from Busemeyer & Bruza (2024) and Roeder et al. (2023).'),
    M('Circuits for the quantum-like families', ['quantum_mind.circuits.builders.order_effects_circuit',
                                                 'quantum_mind.circuits.builders.belief_circuit',
                                                 'quantum_mind.circuits.builders.chsh_circuit'],
      'Every quantum-like family as a Qiskit circuit whose measured frequencies reproduce the model.',
      '''flowchart LR
    m["fitted model"]:::in --> u["subspace unitaries"]:::op --> fl["flag qubits record answers"]:::op --> qc["Qiskit circuit"]:::out''',
      None, None),
    M('Back ends', ['quantum_mind.circuits.backends.run'],
      'Run a circuit on Aer, a fake IBM device, IBM Quantum or Amazon Braket.',
      '''flowchart LR
    qc["circuit"]:::in --> r{"run(qc, backend)"}:::op --> aer["Aer"]:::out & fake["FakeTorino"]:::out & ibm["IBM Quantum"]:::out & br["Amazon Braket"]:::out''',
      None, None),
    M('Bloch-sphere viewer', ['quantum_mind.viz.states.circuit_trajectory', 'quantum_mind.viz.states.belief_trajectory',
                              'quantum_mind.viz.mpl.plot_bloch', 'quantum_mind.viz.interactive.animate_bloch'],
      'Trajectories of qubits through circuits or beliefs, drawn statically or animated.',
      '''flowchart LR
    src["circuit · belief model · state"]:::in --> tr["trajectory of Bloch vectors"]:::op --> st["Matplotlib figure"]:::out
    tr --> an["Plotly animation / HTML"]:::out
    tr --> ent["entanglement and concurrence"]:::out''',
      None, 'Wootters, W. K. (1998). Physical Review Letters, 80, 2245.', '15_bloch_sphere'),
])

# ------------------------------------------------------------------------------------------ origin
#: Origin labels: (badge colour, label, meaning).
LABELS = {
    'unique': ('success', 'Unique to Quantum Mind',
               'designed in this library; it builds on the methods cited with it'),
    'extended': ('info', 'Published model, extended',
                 'implements the cited model as published; Quantum Mind adds the capabilities listed'),
    'published': ('secondary', 'Published method', 'implements the cited method as published'),
    'tool': ('muted', 'Software tool', 'infrastructure rather than a model'),
}

_ALL = ('bootstrap intervals, a per-person posterior updated after every answer, and the choice of the '
        'most informative condition ({doc}`what Quantum Mind adds <../user_guide/extensions>`)')
_BOOT = 'bootstrap intervals (fitted by least squares, so no per-answer updating)'

#: Origin of every atlas entry, by title: (label, what Quantum Mind adds or None).
ORIGIN = {
    # core
    'Lüders rule and answer sequences': ('published', None),
    'Open-system evolution': ('published', None),
    'Model and parameters': ('tool', None),
    'Fit, compare and recover': ('published', None),
    'Individual differences': ('published', None),
    'Uncertainty, online updating and informative questions': ('unique', None),
    # quantum-like families
    'Question order: quantum-like model': ('extended', _ALL),
    'Question order: classical baselines': ('extended', _ALL),
    'Conjunction fallacy': ('extended', _BOOT),
    'Interference and the disjunction effect': ('extended', _ALL),
    'Quantum-like Bayesian network': ('extended', _ALL),
    'Belief dynamics: quantum and Markov walks': ('extended', _ALL),
    'Trust belief (open system)': ('extended', _ALL + '; used for robot hand-over decisions'),
    'Quantum decision theory': ('extended', _ALL),
    'Contextuality tests': ('published', None),
    'Asymmetric similarity': ('extended', _BOOT),
    'Quantum games': ('published', None),
    'Episodic memory overdistribution': ('extended', _BOOT),
    'Concept combination (Fock space)': ('extended', _BOOT),
    'Bistable perception (quantum Zeno)': ('extended', _ALL),
    # robot decision layer
    'Human-model ensemble': ('unique', None),
    'Ask or act': ('unique', None),
    'Human-model service': ('tool', None),
    'Intent resolution': ('unique', None),
    'Recalibration': ('published', None),
    'Conformal prediction sets': ('extended', 'adaptive conformal sets for a person\'s next answer, which keep their '
                                  'long-run coverage while the person changes, and tell the robot when to ask'),
    'Calibration monitor': ('published', None),
    'Confidence gate': ('published', None),
    'Meta-calibrated gate': ('unique', None),
    'Question planner': ('unique', None),
    'Answer models': ('unique', None),
    'Personalised human models': ('extended', 'a full per-person posterior updated after every answer, with credible '
                                  'intervals (method=\'online\')'),
    'Trust-aware hand-over': ('unique', None),
    # perception
    'Perception adapters': ('published', None),
    'Fusion rules': ('extended', 'a quantum-like (Kraus) fusion rule designed here, next to the Bayesian and '
                                 'Dempster-Shafer rules'),
    'Cue incompatibility': ('unique', None),
    'Dwell-time baselines': ('extended', _ALL),
    # learning
    'Clarification environment': ('unique', None),
    'Trust hand-over environment': ('unique', None),
    'Grid world': ('tool', None),
    'Exploration strategies': ('published', None),
    'Tabular agent': ('published', None),
    'Quantum-inspired Q-learning': ('published', None),
    'Variational quantum policy': ('published', None),
    'Tensor-train layers': ('published', None),
    'Quanvolutional filter': ('published', None),
    # quantum
    'State-vector simulator': ('tool', None),
    'Feature maps and ansatz': ('published', None),
    'Variational classifier and regressor': ('published', None),
    'Quantum kernel methods': ('published', None),
    'QAOA': ('published', None),
    'VQE': ('published', None),
    'Grover search': ('published', None),
    'Quantum Fourier transform and period finding': ('published', None),
    'Amplitude estimation': ('published', None),
    'Quantum walks on networks': ('published', None),
    # quantum-inspired
    'QIEA': ('published', None),
    'QPSO': ('published', None),
    'Simulated quantum annealing': ('published', None),
    'MPS classifier': ('published', None),
    'Quantum language model': ('published', None),
    # infrastructure
    'QUBO problems': ('published', None),
    'Trust-aware task allocation': ('unique', None),
    'Circuits for the quantum-like families': ('unique', None),
    'Back ends': ('tool', None),
    'Bloch-sphere viewer': ('tool', None),
}


def badge(title):
    """Sphinx-design badge with the origin label of an atlas entry."""
    colour, label, _ = LABELS[ORIGIN[title][0]]
    return '{bdg-%s}`%s`' % (colour, label)


def origin_counts():
    """Number of atlas entries per origin label."""
    counts = {k: 0 for k in LABELS}
    for _, _, models in PAGES.values():
        for m in models:
            counts[ORIGIN[m['title']][0]] += 1
    return counts


ORDER = ['core', 'quantum_like', 'robotics', 'perception', 'learning', 'quantum', 'inspired', 'infrastructure']

#: Mathematics page for each atlas page.
MATH = {'core': 'core', 'quantum_like': 'quantum_like', 'robotics': 'robotics', 'perception': 'perception',
        'learning': 'learning', 'quantum': 'quantum', 'inspired': 'inspired', 'infrastructure': 'inspired'}


def _role(path):
    """Cross-reference role for an API path (class or function)."""
    mod, _, name = path.rpartition('.')
    try:
        obj = getattr(importlib.import_module(mod), name)
        kind = 'class' if isinstance(obj, type) else 'func'
    except (ImportError, AttributeError):
        kind = 'obj'
    return '{py:%s}`~%s`' % (kind, path)


def write():
    """Write docs/atlas/*.md."""
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for key in ORDER:
        title, intro, models = PAGES[key]
        lines = ['# ' + title, '', intro, 'The equations are on the {doc}`mathematics page <../math/%s>`.' % MATH[key], '',
                 'Legend: blue inputs, violet internal steps, green outputs, coral people, dashed grey '
                 'classical baselines. Each entry is labelled by origin: '
                 + ', '.join('%s %s' % ('{bdg-%s}`%s`' % (c, l), d) for c, l, d in LABELS.values()) + '.', '']
        for m in models:
            total += 1
            first, rest = m['diagram'].split('\n', 1)
            acc = ['    accTitle: Block diagram of %s' % m['title'],
                   '    accDescr: %s' % m['summary'].replace('\n', ' ')]
            kind, adds = ORIGIN[m['title']]
            lines += ['## ' + m['title'], '', badge(m['title']), '', m['summary'], '', '```{mermaid}', first, *acc, rest, STYLE, '```', '']
            lines += ['- **API:** ' + ', '.join(_role(p) for p in m['api'])]
            if m['baselines']:
                lines += ['- **Compared with:** ' + m['baselines']]
            if m['tutorial']:
                lines += ['- **Tutorial:** {doc}`../tutorials/%s`' % m['tutorial']]
            lines += ['- **Origin:** %s: %s.' % (LABELS[kind][1], LABELS[kind][2])]
            if adds:
                lines += ['- **Quantum Mind adds:** ' + adds + '.']
            if m['ref']:
                lines += ['- **Reference:** ' + m['ref']]
            lines += ['']
        (OUT / (key + '.md')).write_text('\n'.join(lines))
    cards = []
    for key in ORDER:
        title, intro, models = PAGES[key]
        cards += [':::{grid-item-card} %s' % title, ':link: %s' % key, ':link-type: doc', '',
                  '%s (%d diagrams)' % (intro, len(models)), ':::', '']
    index = ['# Model atlas', '',
             'A block diagram for every model in Quantum Mind (%d diagrams): what goes in, what happens '
             'inside, what comes out, and which classical baselines it is compared with. Each entry links '
             'to the API, a tutorial where there is one, and a reference.' % total, '',
             '```{mermaid}', '''flowchart LR
    accTitle: How the parts of Quantum Mind fit together
    accDescr: Data from sensors and people feed the core fitting tools, the quantum-like families, perception and learning, which all feed the robot decision layer; quantum and quantum-inspired models feed learning; classical baselines are compared everywhere.
    D[("data · sensors · people")]:::hum --> C["core: fit · compare"]:::op
    C --> QL["quantum-like families"]:::op --> R["robot decision layer"]:::out
    P["perception and fusion"]:::op --> R
    L["learning"]:::op --> R
    QM["quantum models"]:::in --> L
    QI["quantum-inspired"]:::in --> L
    QL --> CI["circuits · back ends · viewer"]:::in
    QM --> CI
    B["classical baselines everywhere"]:::base -.-> QL & P & L & QM & QI''', STYLE, '```', '',
             '## Where each model comes from', '',
             'Every entry carries one of four labels. Most models are published by other researchers and '
             'implemented here from their papers (the reference is on each entry). Quantum Mind\'s own '
             'contributions are the robot decision layer, the environments, the circuits for the '
             'quantum-like families, and the tools that give every published people model uncertainty, '
             'per-person updating and informative-question selection.', '',
             '| Label | Entries | Meaning |', '|---|---|---|']
    counts = origin_counts()
    index += ['| %s | %d | %s |' % ('{bdg-%s}`%s`' % (c, l), counts[k], d) for k, (c, l, d) in LABELS.items()]
    for k in ('unique', 'extended'):
        index += ['', '**%s:** ' % LABELS[k][1] + ', '.join(
            '{doc}`%s <%s>`' % (m['title'], key) for key in ORDER for m in PAGES[key][2] if ORIGIN[m['title']][0] == k) + '.']
    index += ['', '## Pages', '',
             '::::{grid} 1 2 2 2', ':gutter: 3', ''] + cards + ['::::', '', '```{toctree}', ':hidden:', ''] + ORDER + ['```', '']
    (OUT / 'index.md').write_text('\n'.join(index))
    return total


if __name__ == '__main__':
    print(write(), 'diagrams')
