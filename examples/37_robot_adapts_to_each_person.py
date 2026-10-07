"""Robot that adapts to each person
================================

Robotics, human-robot interaction: one workstation, several people, one robot. The robot (1) learns
each person's answering from the first few answers while keeping a guaranteed miss rate on its
predictions, (2) gives robot-assisted tasks to the people who trust it, and (3) is trained against a
population of simulated people rather than one. All people are simulated from the library's models;
no real participants are involved.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_37.png'
import numpy as np
from quantum_mind import fit
from quantum_mind.applications.allocation import trust_aware_allocation, expected_costs, decode
from quantum_mind.applications.calibration import AdaptiveConformalSets
from quantum_mind.applications.personalisation import PopulationPrior
from quantum_mind.envs import TrustHandoverEnv, sample_people
from quantum_mind.families.dynamics import OpenSystemBelief
from quantum_mind.families.order_effects import QuantumOrderModel4D
from quantum_mind.inspired.exploration import TabularAgent, EpsilonGreedy, run_episodes

rng = np.random.default_rng(0)

# %%
# 1. Learn one person while keeping a guaranteed miss rate
# --------------------------------------------------------
# A group of earlier (simulated) people gives the population fit. A new person answers two questions
# in both orders, and differs from the group. The per-person posterior adapts after every answer;
# adaptive conformal sets keep the long-run miss rate near 15% whatever the person does, and the robot
# asks for confirmation whenever more than one answer pair is still in the set. When no single answer
# pair is likely enough for the target, the set grows and the robot asks; it acts alone only when one
# answer pair carries enough probability.
group_model = QuantumOrderModel4D(t1=0.5, t2=1.0, t3=1.0, phi=0.6)
group = fit(QuantumOrderModel4D, group_model.sample(None, 300, rng), restarts=4)
person_truth = QuantumOrderModel4D(t1=0.1, t2=0.9, t3=1.4, phi=0.36).predict(None)

prior = PopulationPrior(QuantumOrderModel4D, mean=group.model.to_vector(), cov=np.eye(4) * 0.3)
person = prior.online(n_particles=300, rng=np.random.default_rng(1))
sets = AdaptiveConformalSets(alpha=0.15, gamma=0.05)
asks, sizes = 0, []
for t in range(150):
    order = 'AB' if t % 2 else 'BA'
    p = person.predict()[order]
    asks += sets.should_ask(p)
    sizes.append(int(sets.predict_set(p).sum()))
    answer = int(rng.choice(4, p=person_truth[order]))
    sets.update(p, answer)
    person.update(order, answer)
print('miss rate %.3f (target 0.150, guaranteed within %.3f of it)' % (sets.miscoverage, sets.bound()))
print('asked for confirmation on %d of 150 answers; mean set size %.2f of 4' % (asks, np.mean(sizes)))
print('predicted AB answers %s, true %s' % (np.round(person.predict()['AB'], 2), np.round(person_truth['AB'], 2)))

# %%
# 2. Give robot-assisted tasks to the people who trust the robot
# --------------------------------------------------------------
# Three people with different initial trust (trust-qubit angle phi0; P(trust) = cos^2(phi0 / 2)) and
# four tasks, two of which need the robot's help. The trust-aware plan is compared with a plan that
# ignores trust, by expected cost under the people's true trust.
people = [OpenSystemBelief(phi0=a) for a in (0.4, 1.6, 2.6)]
trust = np.array([np.cos(m.phi0 / 2) ** 2 for m in people])
effort = np.array([[1.0, 1.2, 1.0, 1.1], [0.9, 1.0, 1.1, 1.0], [0.8, 0.9, 1.0, 0.9]])
reliance = np.array([1.0, 0.8, 0.0, 0.0])
true_cost = expected_costs(effort, trust, reliance, fail_cost=10)
aware, _ = trust_aware_allocation(effort, trust, reliance, fail_cost=10, workload=0.3).brute_force()
blind, _ = trust_aware_allocation(effort, [1, 1, 1], reliance, fail_cost=10, workload=0.3).brute_force()
for name, x in (('trust-aware', aware), ('trust-blind', blind)):
    X = np.asarray(x).reshape(3, 4)
    print('%-11s %s  expected cost %.2f' % (name, decode(x, 3, 4), float((true_cost * X).sum())))

# %%
# 3. Train against one simulated person or a population
# -----------------------------------------------------
# A tabular agent learns when to hand over, hand over slowly, ask or wait. One agent trains against
# a single simulated person, another against a new person drawn every episode from the population
# prior; both are evaluated on 200 new people. In this environment asking collapses every person's
# trust in the same way, so the two often learn the same policy: the population is a safeguard, not
# a guaranteed gain, and the numbers below are what this run measured.
base = OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3)
population = PopulationPrior(OpenSystemBelief, mean=base.to_vector(), cov=np.diag([1.0, 0.5, 0.5, 0.5]))


def state(o):
    return ((int(o[0]) + 1) * 3 + int(o[1]) + 1) * 4 + min(int(o[2] * 4), 3)


def train(env, seed):
    agent = TabularAgent(36, 4, EpsilonGreedy(0.3, decay=0.999, seed=seed))
    run_episodes(agent, env, 1500, state_fn=state, seed=seed)
    return agent


def evaluate(agent, test):
    total = []
    for i, p in enumerate(test):
        env = TrustHandoverEnv(model=p)
        o, _ = env.reset(seed=10_000 + i)
        r_sum = 0.0
        for _ in range(env.n_steps):
            o, r, terminated, truncated, _ = env.step(agent.greedy(state(o)))
            r_sum += r
        total.append(r_sum)
    return float(np.mean(total))


test_people = sample_people(population, 200, np.random.default_rng(99))
one = evaluate(train(TrustHandoverEnv(model=base), 0), test_people)
pop = evaluate(train(TrustHandoverEnv(people=sample_people(population, 300, np.random.default_rng(0))), 0), test_people)
print('mean return on new people: trained on one person %.2f, on a population %.2f' % (one, pop))
