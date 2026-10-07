# Robotics

A robot is a machine that **senses** its surroundings, **decides** what to do and **acts** on the
world, over and over, in real time. Everything else in robotics, from motors to machine learning,
serves one of those three steps.

```{mermaid}
flowchart LR
    accTitle: Robotics, diagram 1
    accDescr: The sense, perceive, decide and act loop: sensors observe the world and people, perception estimates the situation, the robot decides, actuators change the world.
    W(("world<br/>and people")):::hum -->|light, sound, contact| S["Sense<br/>cameras · lidar · microphones · joint encoders"]:::in
    S --> P["Perceive<br/>where am I, what is there, who is there"]:::op
    P --> D["Decide<br/>plan · ask · act · wait"]:::op
    D --> A["Act<br/>motors · grippers · speech"]:::out
    A -->|changes| W
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef hum fill:#3d1414,stroke:#ff7b72,color:#e6edf3
```

## From the basics

Robot
: A machine with sensors, a computer and actuators that carries out tasks with some autonomy. Arms in
  factories, mobile robots in warehouses, drones, humanoids and assistive robots are all robots.

Degrees of freedom (DoF)
: The number of independent ways a robot can move. A rigid object in space has 6 (three positions,
  three rotations); a typical arm has 6 or 7 joints so its gripper can reach any pose in its workspace.

Kinematics
: The geometry of motion, ignoring forces. **Forward kinematics** maps joint angles to the gripper's
  position; **inverse kinematics** finds joint angles that put the gripper where you want it.

Dynamics and control
: Dynamics relates forces and torques to motion. A **controller** (for example PID: proportional,
  integral, derivative) turns the error between where the robot is and where it should be into motor
  commands, hundreds or thousands of times per second.

State estimation
: Sensors are noisy, so the robot keeps a **belief**, a probability distribution over its state, and
  updates it with every measurement (Kalman filter, particle filter). **SLAM** (simultaneous
  localisation and mapping) builds a map while locating the robot in it.

Planning
: Finding a sequence of actions or a path from start to goal: graph search (A\*), sampling-based
  planners (RRT, PRM) and, when outcomes are uncertain, decision-theoretic planning (MDPs and POMDPs).

Human-robot interaction (HRI)
: The study of robots that work with people: communication, shared control, safety and **trust**. A
  robot that hands a tool to a person must predict what the person wants and how much they trust it.

Middleware
: Software that connects the parts. **ROS 2** (Robot Operating System) passes messages between nodes
  (camera driver, detector, planner, controller) over topics and services.

## A small example: forward kinematics of a two-link arm

```python
import numpy as np

def forward_kinematics(theta1, theta2, l1=0.5, l2=0.4):
    """Position of the gripper of a planar arm with two joints."""
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return round(float(x), 3), round(float(y), 3)

print(forward_kinematics(np.pi / 4, np.pi / 6))   # (0.457, 0.74)
assert forward_kinematics(np.pi / 4, np.pi / 6) == (0.457, 0.74)
```

## A small example: a Bayes filter in one line

The robot believes the door is open with probability 0.5. Its sensor says "open" correctly 80% of the
time and wrongly 30% of the time. After one "open" reading:

```python
prior = 0.5
p_open_given_reading = 0.8 * prior / (0.8 * prior + 0.3 * (1 - prior))
print(round(p_open_given_reading, 3))   # 0.727
assert round(p_open_given_reading, 3) == 0.727
```

## Where Quantum Mind fits

Quantum Mind works in the **Decide** box, where the robot has to reason about people. Classical
robotics assumes a person's answers are fixed facts; in practice asking a question can change the
answer to the next one, and asking about trust can change trust. The library gives the robot models
of those effects, always next to the classical models, plus calibrated confidence, question planning
and hand-over policies.

- [Tutorial: when to ask for help](../tutorials/04_ask_or_act.ipynb)
- [Tutorial: trust-aware hand-over](../tutorials/03_trust_on_a_qubit.ipynb)
- [Model atlas: robot decision layer](../atlas/robotics.md)
- [ROS 2 integration](../user_guide/ros2.md)

## References

- Siciliano, B., & Khatib, O. (Eds.). (2016). *Springer Handbook of Robotics* (2nd ed.). Springer.
- Lynch, K. M., & Park, F. C. (2017). *Modern Robotics: Mechanics, Planning, and Control*. Cambridge
  University Press.
- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
- Corke, P. (2023). *Robotics, Vision and Control* (3rd ed.). Springer.
- Goodrich, M. A., & Schultz, A. C. (2007). Human-robot interaction: A survey. *Foundations and Trends
  in Human-Computer Interaction*, 1(3), 203-275.
- Macenski, S., Foote, T., Gerkey, B., Lalancette, C., & Woodall, W. (2022). Robot Operating System 2:
  Design, architecture, and uses in the wild. *Science Robotics*, 7(66), eabm6074.
