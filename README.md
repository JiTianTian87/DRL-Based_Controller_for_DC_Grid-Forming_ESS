# Deep Reinforcement Learning Based Robust Control for DC Grid-Forming Energy Storage Systems


This repository contains code and materials related to our conference paper:

**"Deep Reinforcement Learning Based Robust Control Strategy for Power Sharing and Voltage Regulation in DC Grid-Forming Energy Storage System"**  
Tiantian Ji, Pengfeng Lin, Miao Zhu, Qingzuo Meng, Chuanlin Zhang, Jiebei Zhu.

---

## Abstract

Grid-forming energy storage systems (ESSs) play a crucial role in modern DC power systems, enabling stable power sharing and voltage regulation in the presence of intermittent renewable energy sources. Constant power loads (CPLs) introduce negative impedance characteristics, challenging system stability.  

We propose a **model-free deep reinforcement learning (DRL) controller** for a grid-forming ESS, transforming the control problem into a sequential decision-making problem via a **Markov Decision Process (MDP)**. The controller:

- Adapts to varying system conditions without prior modeling knowledge.
- Maintains large-signal stability under CPL disturbances.
- Provides rapid dynamic response and accurate voltage/current tracking.

Simulation results demonstrate superior performance over traditional double-loop PI controllers in terms of voltage regulation and current sharing.

---

## Features

- **Data-driven DRL controller** based on the Double DQN algorithm.
- Handles **step changes in load (CPL)** and **reference voltage (Vref)** effectively.
- Robust to **system parameter uncertainties** (inductor and capacitor deviations).
- Simulation implemented in **MATLAB/Simulink**.

---

## System Model

The study considers a DC grid-forming ESS with:

- Parallel Buck converters supplying power.
- CPLs modeled as current sources \(i_{CPL} = P_{CPL}/v_{bus}\).
- State space includes voltage and current errors, delayed and differential signals.
- Action space: duty cycles of the converters.
- Reward function designed to optimize voltage regulation and current sharing.


---

## Results

- **Faster voltage tracking** than double-loop PI controller (2-4ms response time).  
- **Stable current sharing** among parallel modules.  
- **Robustness**: Handles ±20% deviation in inductance/capacitance while maintaining stability.  

Simulation figures from the paper:

- Voltage response under CPL step changes
- Power sharing among parallel converters
- Voltage response under Vref changes
- Voltage response under parameter uncertainty

---

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
