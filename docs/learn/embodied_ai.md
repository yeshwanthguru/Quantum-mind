# Embodied AI

Embodied AI studies intelligence that has a **body**: agents that perceive and act in a physical (or
simulated physical) world, where every action changes what they will perceive next. A chatbot answers
text; an embodied agent must find the cup, reach it, hand it over and notice whether the person took
it. Robotics supplies the body; AI, ML and RL supply the learning; and when people are involved, the
agent also needs models of **them**.

```{mermaid}
flowchart TB
    subgraph Body["body"]
      SEN["sensors"]:::in
      ACT["actuators"]:::out
    end
    subgraph Brain["agent"]
      PER["perception<br/>vision · language · touch"]:::op --> MEM["world model<br/>objects · places · people"]:::op
      MEM --> POL["policy / planner<br/>learned or designed"]:::op
      HUM["models of people<br/>intent · trust · order effects"]:::hum --> POL
    end
    SEN --> PER
    POL --> ACT
    ACT -->|changes what is sensed next| SEN
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef hum fill:#3d1414,stroke:#ff7b72,color:#e6edf3
```

## From the basics

Embodiment
: The idea that intelligence depends on having a body in an environment: what an agent can sense and
  do shapes what it needs to compute. Brooks argued in 1991 that robots should be built from layers of
  behaviour coupled directly to the world rather than from a single central model.

Simulation and sim-to-real
: Embodied agents are usually trained in simulators (Isaac Sim, MuJoCo, Habitat, Gazebo) and then
  transferred to real robots. The **sim-to-real gap**, the difference between simulated and real
  physics and sensors, is a main obstacle. The same gap exists for people: simulated users rarely
  behave like real ones.

Embodied tasks
: Navigation ("go to the kitchen"), object search, manipulation ("put the cup in the sink"),
  instruction following and **interactive** tasks where the agent asks questions or hands objects to a
  person.

Foundation models for robots
: Large models trained on web data and robot data that map images and instructions to actions
  (vision-language-action models). They make robots more general but give uncalibrated confidence and
  can be sensitive to how a request is phrased.

Behaviour trees
: A common way to structure robot behaviour from reusable nodes (sequence, fallback, condition,
  action). Decision rules such as "ask or act" fit naturally as condition nodes.

Interaction loops with people
: In interactive tasks the person is part of the environment, and the agent's own questions change
  the person's state. This is where quantum-like models of judgement and trust matter: they predict
  that the order and the act of asking have effects.

## A small example: one step of an interactive embodied task

```python
from quantum_mind.envs import ClarificationEnv
env = ClarificationEnv()                 # a simulated person with an ambiguous request
obs, info = env.reset(seed=3)
obs, reward, done, truncated, info = env.step(0)     # ask question 0
print(obs, reward, done)
obs, reward, done, truncated, info = env.step(env.Q) # act on hypothesis 0
print(reward, done)
```

## Where Quantum Mind fits

Quantum Mind gives an embodied agent the **models of people** box: answer models with order effects,
trust that responds to being asked, an ensemble that knows when its models disagree, decision rules
that use all of this, Gymnasium environments to train against, and a ROS 2 node to run it on a robot.

- [Tutorial: environments and exploration](../tutorials/10_rl_environments.ipynb)
- [Tutorial: trust on a qubit](../tutorials/03_trust_on_a_qubit.ipynb)
- [ROS 2 integration](../user_guide/ros2.md)

## References

- Pfeifer, R., & Bongard, J. (2006). *How the Body Shapes the Way We Think: A New View of
  Intelligence*. MIT Press.
- Brooks, R. A. (1991). Intelligence without representation. *Artificial Intelligence*, 47(1-3),
  139-159.
- Duan, J., Yu, S., Tan, H. L., Zhu, H., & Tan, C. (2022). A survey of embodied AI: From simulators to
  research tasks. *IEEE Transactions on Emerging Topics in Computational Intelligence*, 6(2), 230-244.
- Brohan, A., et al. (2023). RT-2: Vision-language-action models transfer web knowledge to robotic
  control. arXiv:2307.15818.
- Colledanchise, M., & Ögren, P. (2018). *Behavior Trees in Robotics and AI: An Introduction*. CRC Press.
- Bartneck, C., Belpaeme, T., Eyssel, F., Kanda, T., Keijsers, M., & Šabanović, S. (2020).
  *Human-Robot Interaction: An Introduction*. Cambridge University Press.
