# Reinforcement Learning Coursework

This repository contains my submission for the **Reinforcement Learning 2024/2025 coursework** at the University of Edinburgh.  
The project covers implementations of reinforcement learning algorithms ranging from **dynamic programming** methods to **deep reinforcement learning for continuous control**.

---

## 🚀 Project Overview

The coursework was structured into four main exercises, each focusing on a different class of RL algorithms:

1. **Dynamic Programming (Exercise 1)**  
   - Implemented **Value Iteration** and **Policy Iteration**.  
   - Tested on a custom MDP example (“Frog on a Rock”).  

2. **Tabular Reinforcement Learning (Exercise 2)**  
   - Implemented **Q-Learning** and **Every-Visit Monte Carlo**.  
   - Applied to **FrozenLake8x8-v1** (both slippery and non-slippery variants).  
   - Analysed the effect of hyperparameters such as ε and γ.  

3. **Deep Reinforcement Learning (Exercise 3)**  
   - Implemented a **Deep Q-Network (DQN)** with replay buffer and target network.  
   - Compared performance against a discrete tabular solution.  
   - Experiments run on **MountainCar-v0**, with additional tests on **CartPole**.  
   - Explored different **ε-scheduling strategies** and analysed DQN loss behaviour.  

4. **Continuous Control (Exercise 4)**  
   - Implemented **Deep Deterministic Policy Gradient (DDPG)** for continuous action spaces.  
   - Applied to the **Highway-Env Racetrack** task.  
   - Tuned network sizes and hyperparameters to achieve reliable performance.  

---

## 🗂 Repository Structure

```text
rl2025/
├── exercise1/          # Dynamic Programming (Value Iteration, Policy Iteration)
├── exercise2/          # Tabular RL (Q-Learning, Monte Carlo)
├── exercise3/          # Deep Q-Networks and discrete comparison
├── exercise4/          # DDPG for continuous control
├── util/               # Utilities for logging and result processing
├── answer_sheet.py     # Short written answers to analysis questions
└── constants.py        # Configuration parameters
```

---

## Results Summary

* **Dynamic Programming**: Correct policies computed for small MDPs.
* Key aspects such as convergence behaviour, hyperparameter effects, and exploration strategies were investigated.
* Tabular methods were compared, as well as deep methods, on their respective environments.
* **Racetrack (DDPG)**: Achieved evaluation returns above the required performance threshold (over 500) after tuning hidden layer sizes.

---

## Notes

* This repository was developed as part of an **assessed coursework** and is **not prepared for general reuse**.
* The implementations follow the assignment specification closely and are structured according to the provided starter code.
* Performance results and answers to conceptual questions are included in `answer_sheet.py`.

