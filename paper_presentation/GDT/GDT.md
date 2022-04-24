---
marp: true
theme: default
paginate: true
# _class: invert
# color: white
class: lead
style: |
  h1 {
    color: #3d3d3d
  }
# style: |
#   section {
#     background-color: #ffffff;
#   }
#   h1 {
#     font-size: 50px;
#     color: #2a2a2a;
#   }
---

# Neural Kernel Without Tangents

**ICML'20 Citation: 37**

Vaishaal Shankar, Alex Fang, Wenshuo Guo,  Sara Fridovich-Keil, Ludwig Schmidt, Jonathan Ragan-Kelley, Benjamin Recht

UC Berkeley, MIT


---

Propose single architecture solution for the following problem

- NIPS'17 [Hindsight experience replay](https://papers.nips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html) (By Marcin Andrychowicz)

  - Sparse reward tasks are hard to learn
  - Relabel the trajectory to the tasks that have been achieved in the trajectory (kind of intrinsic reward or curricular learning) 
  - But relabel the trajectory is still too sparse, so let the task be the state where the agent will achieve after several steps
  - Put the relabeled trajectories into replay buffer
  
---

- Multi-Task: NIPS'20 [Generalized Hindsight for Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/57e5cb96e22546001f1d6520ff11d9ba-Paper.pdf) (By Alexander C. Li)

  - Follow the previous work, address the drawbacks of HER
  
    - The reward function maybe complex, state may not be a good label, EX: Atari Pong
    - How to find a better label? Is there a more general way to relabel the trajectory?
  - Find the label(task) on which our current trajectory does better than the other ones.
  - 

---

- Meta Imitation Learning: NIPS'17: One-Shot Imitation Learning (By Yan Duan)
- Offline RL: NIPS'21: Decision transformer: Reinforcement learning via sequence modeling (By Lili Chen)
- Budget-aware RL

---

