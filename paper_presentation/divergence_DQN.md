---
marp: true
theme: default
---

# Towards Characterizing Divergence in Deep Q-Learning

arXiv 2019

Citation: 40

OpenAI, UCB

*Joshua Achiam, Ethan Knight, Pieter Abbeel*

---
# Motivation
The **Deadly Triad** of DQN:

Once we put **"bootstrapping"**, **"off-policy learning"**, **"function approximation"** together, they will lead to **divergence** in DQN.

However,  the conditions under which divergence occurs are not
well-understood.

---

# Main Ideas
Why dose DQN diverge under deadly triad?
How about analyzing DQN with NTK?

---
# The Result of Analyzation

- The main reason why DQN diverge is **Over-generalization** and **improper(too large or too small) learning rate**.

---
# Outline

- Motivation
- Main Ideas 
- The Result of Analyzation
- Analyzation Setup
- Building Intuition for Divergen with NTK
- PreQN
- Experiments

---

# Analyzation Setup

## Contraction Map
Let $X$ be a vector space with norm $k \cdot k$, and $f$ a function from $X$ to $X$. If $\forall x, y \in X$, $f$ satisfies

$$||f(x) − f(y)|| \leq \beta ||x − y|| \ \ \ $$

with $\beta \in [0, 1)$, then $f$ is called a contraction map with modulus $\beta$

---

## Banach Fixed-Point Theorem
Let $f$ be a contraction map, $\exist x_u \ \ st \ \ f(x_u) = x_u$. 

### Properties

- $x_u$ is an unique fixed-point. 

- Because $f$ is a contraction map, $x_u$ can be obtained by the repeated application of $f$: for any point $x_0 \in X$, if we define a sequence of points $\{ x_n \}$ such that $x_n = f(x_n − 1)$, $\lim_{n \to \infty} x_n = x$.

---

## Bellmen Operator & Q-Function
Let $Q(s, a)$ be the Q function and $Q^*(s, a)$ be the optimal Q function.

---
# NTK of DQN

The Bellman quation of DQN with the experience distribution $\rho$ in replay buffer

$$Q_{k+1}(s, a) = E_{s, a \sim \rho}[Q_k(s, a) + \alpha_k (\hat{\tau}^* Q_k(s, a) − Q_k(s, a))]$$

$$\hat{\tau}^{*} = Q_k(s, a) = r + \gamma \ max_{a'} Q_k(s', a')$$

The TD error $\delta_t$

$$\delta_t = \tau^* Q(s_t, a_t) − Q(s_t, a_t)
= r_t + \gamma \ \mathop{\max}_{a'} \ Q(s_{t+1}, a') − Q(s_t, a_t)$$

Update the weights

$$\theta' = \theta + \alpha E_{s, a \sim \rho}[(\tau^* Q_{\theta}(s, a) − Q_{\theta}(s, a)) \ \nabla_{\theta} Q_{\theta}(s, a)]$$

---
# NTK of DQN

The **Taylor Expansion** of $Q$ around $\theta$ at a state-action pair $(\bar{s}, \bar{a})$. 

$$Q_{\theta'} (\bar{s}, \bar{a}) = Q_{\theta}(\bar{s}, \bar{a})+\nabla_{\theta}Q_{\theta}(\bar{s}, \bar{a})^{\top}(\theta'−\theta)$$

Combine with

$$\theta' - \theta = \alpha E_{s, a \sim \rho}[(\tau^* Q_{\theta}(s, a) − Q_{\theta}(s, a)) \ \nabla_{\theta} Q_{\theta}(s, a)]$$

We get

$$Q_{\theta'} (\bar{s}, \bar{a}) = Q_{\theta}(\bar{s}, \bar{a}) + \alpha E_{s, a \sim \rho}[k_{\theta}(\bar{s}, \bar{a}, s, a) (\tau^*Q_{\theta}(s, a) − Q_{\theta}(s, a))]$$

$$k_{\theta}(\bar{s}, \bar{a}, s, a) = \nabla_{\theta}Q_{\theta}(\bar{s}, \bar{a})^{\top} \nabla_{\theta} Q_{\theta}(s, a)$$

Where $k_{\theta}(\bar{s}, \bar{a}, s, a)$ is **NTK** 