# Deep Reinforcement Learning

Paper List

## Overview & Supplement

### Agent57- Outperforming the human Atari benchmark: 

Deepmind的work，新的東西不多，但是在Blog整理了近五年RL的重大發展，可以當作一個Overview快速了解RL的發展

[Blog](https://deepmind.com/blog/article/Agent57-Outperforming-the-human-Atari-benchmark)

### 当我们在谈论 DRL：从AC、PG 到 A3C、DDPG:

把Actor-Critic、Policy Gradient、DDPG之間的脈絡梳理得很清楚，剛上完老師的RL可以來看，剛好可以補充一些老師沒整理的東西。

[Blog](https://zhuanlan.zhihu.com/p/36506567)

### Policy Gradient Algorithms: 

在Blog裡面整理了許多常見的Policy Gradient算法，包含各算法的推導

[Paper](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)

## Real-World Challenges


### Challenges of Real-World Reinforcement Learning
### Learning to Adapt in Dynamic, Real-World Environments via Meta-Reinforcement Learning

## Open-Ended

用這個關鍵字找到的論文很少，如果是針對Multi-Task的話，主要都找到Curriculum Learning居多


### POET: Endlessly Generating Increasingly Complex and Diverse Learning Environments and their Solutions through the Paired Open-Ended Trailblazer

實驗室報過，

[Paper](https://eng.uber.com/poet-open-ended-deep-learning/)


## Curriculum Learning with RL

Curriculum Learning即是設計一套循序漸進或有多個不同目標的任務給Agent訓練，讓Agent可以較容易收斂或是能完成多個目標，此種方法也常常用在Open-Ended RL，讓Agent可以更Generalize在多個Tasks。

### [Remy Portelas et al] Automatic Curriculum Learning For Deep RL: A Short Survey

整理了很多Curriculum Learning的方法，但是比較雜且沒有系統，可以先讀Blog再來這邊找相關論文。

[Paper](https://arxiv.org/abs/2003.04664)


### Curriculum for Reinforcement Learning:

Blog，但整理很多用於多任務學習的方法，建議可以當作Overview來讀。

[Blog](https://lilianweng.github.io/lil-log/2020/01/29/curriculum-for-reinforcement-learning.html#curriculum-through-self-play)

### [Schaul et al., 2015a] Tom Schaul et al. Universal value function approximators. In ICML

鼎鼎大名David Silver發的，Citation 401，把不同任務的Goal也當作Value Function的參數，使之可以針對不同任務一次優化。

[Paper](http://proceedings.mlr.press/v37/schaul15.pdf)

### [Florensa et al., 2018] Carlos Florensa et al. Automatic goal generation for reinforcement learning agents. In ICML

用GAN自動產生Curriculum給Agent訓練。

[Paper](https://arxiv.org/pdf/1705.06366.pdf)

### [Jabri et al., 2019] Unsupervised curricula for visual meta-reinforcement learning. In NeurIPS.

看不懂QAQ，但是應該有用。

[Paper](https://arxiv.org/abs/1912.04226)

### [Colas et al., 2019] Cedric Colas et al. CURIOUS: Intrinsically motivated modular multi-goal reinforcement learning. In ICML.

修改了UVFA(Universal Value Function Approximation)，使之可以透過輸入參數模組化。

[Paper](https://arxiv.org/abs/1810.06284)

## Curiosity

### Curiosity-driven Exploration by Self-supervised Prediction: 

提出Intrinsic Curiosity Module的原Paper，旨在讓ICM提供額外的reward，在Agent探索沒看過的state時，ICM會提供較大的reward，若是在已經看過的state，則會提供較小的reward。

[Paper](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)

### Exploration by Random Network Distillation

之前實驗室有報過，旨在解決ICM會一直花時間在一些不重要但會一直改變的東西上

[Paper](https://arxiv.org/abs/1810.12894)

## Memory & World Model

### World Model

這篇提出一個嶄新的概念，以往的RL 都是用MDP model environment，然後將MPD 寫成Q function的 Recurrence 再用Value 或 Policy Iteration based的方式解。但World Model提出一個新的觀點，如果先訓練一個RNN可以依據當前的State和Action去Predict 下一個State和當前Action的Reward，再利用這種預測的能力來讓Agent行動呢？而實驗結果相當好，只需要一個RNN和一個矩陣相乘就可以跟傳統RL相匹敵。

[Paper](https://arxiv.org/abs/1803.10122)

## PlaNet

World Model的改良版

[Paper](https://arxiv.org/pdf/1811.04551.pdf)

[Blog](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html)

### Dreamer: 

PlaNet的改良版

[Paper](https://arxiv.org/pdf/1912.01603.pdf)

[Blog](https://ai.googleblog.com/2020/03/introducing-dreamer-scalable.html)

## Meta-Learning

### Discovering Reinforcement Learning Algorithms

提出一種meta-learning的框架讓agent可以從data自行適應學習

[Paper](./rl_algos/Discovering%20Reinforcement%20Learning%20Algorithms.pdf)


## Alternative

### Evaluating Agents without Rewards

[Paper](rl_algos/Evaluating%20Agents%20without%20Rewards.pdf)

# Reference

- [UCB CS285 Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [UM IFT 6085: Theoretical principles for deep learning](http://mitliagkas.github.io/ift6085-dl-theory-class/)
- [Waterloo University CS885 Spring: Reinforcement Learning](https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring18/schedule.htmls)
- [Caltech CS 159: Data-Driven Algorithm Design](https://sites.google.com/view/cs-159-spring-2020/)