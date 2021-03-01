---
marp: true
theme: default
paginate: true
# _class: invert
# color: white
# backgroundColor: black
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

However, **the conditions under which divergence occurs are not well-understood.**

---

# Main Ideas
Why dose DQN diverge under deadly triad?
How about analyzing DQN with NTK?

---
# The Result of Analyzation

- The main reason why DQN diverge is **Over-generalization** and **improper(too large or too small) learning rate**.
- The **network architecture seems to affect the convergence of DQN**

---
# Outline

- Motivation
- Main Ideas 
- The Result of Analyzation
- Analyzation Setup
- NTK of DQN
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

Thus, the Q-values before and after an update are related by:

$$Q_{\theta'} (\bar{s}, \bar{a}) = Q_{\theta}(\bar{s}, \bar{a}) + \alpha E_{s, a \sim \rho}[k_{\theta}(\bar{s}, \bar{a}, s, a) (\tau^*Q_{\theta}(s, a) − Q_{\theta}(s, a))]$$

$$k_{\theta}(\bar{s}, \bar{a}, s, a) = \nabla_{\theta}Q_{\theta}(\bar{s}, \bar{a})^{\top} \nabla_{\theta} Q_{\theta}(s, a) \qquad \qquad \tag{9}$$

Where $k_{\theta}(\bar{s}, \bar{a}, s, a)$ is **NTK** 

---
# Building Intuition for Divergen with NTK

## Theorem 1
The Q function is represented as a vector in $\mathbb{R}^{|S||A|}$, and the Q-values before and after an update are related by:

$$Q_{\theta'} = Q_{\theta} + \alpha K_{\theta} D_{\rho}(\tau^* Q_{\theta} − Q_{\theta}) \qquad \qquad \tag{10}$$


where $K_{\theta} \in \mathbb{R}^{|S||A| × |S||A|}$ is the matrix of entries given by the NTK $k_{\theta}(\bar{s}, \bar{a}, s, a)$, and $D_{\rho}$ is a  matrix with entries given by $\rho(s, a)$, the distribution from the replay buffer.

---
Consider the operator $\mathcal{U}_3$ given by

$$\mathcal{U}_3 Q = Q + \alpha K D_{\rho} (\tau^* Q − Q) \qquad \qquad \tag{14}$$

## Lemma 3

Under the same conditions as Theorem 1, the Q-values before and after an update are related by 

$$Q_{\theta} = \mathcal{U}_3 Q_{\theta} \qquad \qquad \tag{15}$$

---
## Theorem 2
Let indices $i, j$ refer to state-action pairs. **Suppose** that $K$ and $\rho$ satisfy the conditions:

$$\forall i, \ \alpha K_{ii}\rho_{i} < 1 \qquad \qquad \tag{16}$$

$$\forall i, \ (1 + \gamma)\sum_{j \not ={i}} |K_{ij}|\rho_{j} \leq (1 − \gamma)K_{ii} \rho_{i} \qquad \qquad \tag{17}$$

Then $\mathcal{U}_3$ is a contraction on $Q$ in the sup norm, with fixedpoint $Q^*$.

---
### Proof of Theorem 2
$$
[\mathcal{U}_3 Q_1 − \mathcal{U}_3 Q_2]_i = [(Q_1 + \alpha K D_{\rho} (\tau^* Q_1 − Q_1)) - (Q_2 + \alpha K D_{\rho} (\tau^* Q_2 − Q_2))]_{i}
$$

$$
= [(Q_1 − Q_2) + \alpha K D_{\rho}((\tau^* Q_1 − Q_1) - (\tau^* Q_2 − Q_2))]_i
$$

$$ 
= \sum_j \delta_{ij} [Q_1 − Q_2]_j + \alpha \sum_j K_{ij} \rho_j [(\tau^* Q_1 − Q_1) − (\tau^* Q_2 − Q_2)]_j
$$

$$
= \sum_j \ (\delta_{ij} − \alpha K_{ij} \rho_j ) [Q_1 − Q_2]_j + \alpha \sum_j K_{ij} \rho_j [\tau^* Q_1 − \tau^* Q_2]_j
$$

$$
\leq
\sum_j (|\delta_{ij} − \alpha K_{ij} \rho_j| + \alpha \gamma |K_{ij}| \rho_j) ||Q_1 − Q_2||_{\infty}
$$

Thus we can obtain a modulus as $\beta(K) = \mathop{max}_i \ \sum_j (|\delta_{ij} − \alpha K_{ij} \rho_j| + \alpha \gamma |K_{ij}| \rho_j)$

---

We’ll break it up into on-diagonal and off-diagonal parts, and assume that $\alpha K_{ii} \rho_{i} \le 1$:

$$
\beta(K) = \mathop{max}_i \ \sum_j (|\delta_{ij} − \alpha K_{ij} \rho_j| + \alpha \gamma |K_{ij}| \rho_j)
$$

$$
= \mathop{max}_i \ ((|1 − \alpha K_{ii} \rho_{i}| + \alpha \gamma K_{ii} \rho_i) + (1 + \gamma) \alpha \sum_{j \not ={i}} |K_{ij}| \rho_j)
$$

$$
= \mathop{max}_i \ ((1 − \alpha K_{ii} \rho_{i} + \alpha \gamma K_{ii} \rho_i) + (1 + \gamma) \alpha \sum_{j \not ={i}} |K_{ij}| \rho_j)
$$

$$
= \mathop{max}_i \ (1 − (1 - \gamma) \alpha K_{ii} \rho_{i} + (1 + \gamma) \alpha \sum_{j \not ={i}} |K_{ij}| \rho_j)
$$

According to Banach Fixed-Point Theorem, if $\beta(K) < 1$, $[\mathcal{U}_3 Q_1 − \mathcal{U}_3 Q_2]_i$ would converge

---

Thus,

$$
\forall i, \ \beta(K) < 1
$$

$$
\forall i, \ \mathop{max}_i \ (1 − (1 - \gamma) \alpha K_{ii} \rho_{i} + (1 + \gamma) \alpha \sum_{j \not ={i}}
|K_{ij}| \rho_j) < 1
$$

$$
\forall i, \ 1 − (1 - \gamma) \alpha K_{ii} \rho_{i} + (1 + \gamma) \alpha \sum_{j \not ={i}} |K_{ij}| \rho_j < 1
$$

$$
\forall i, \ (1 + \gamma) \sum_{j \not ={i}} |K_{ij}| \rho_j < (1 - \gamma) K_{ii} \rho_{i}
$$

$$
\forall i, \ \frac{(1 + \gamma)}{(1 - \gamma)} \sum_{j \not ={i}} |K_{ij}| \rho_j < K_{ii} \rho_{i}
$$

Note that this is a quite restrictive condition, since for $\gamma$ high (EX: 0.99), 
$(1 + \gamma)/(1 − \gamma)$ will be quite large, and the left hand side has a sum over all off-diagonal terms in a row.


---
## Intuition 3
- The stability of Q-learning is **tied to the generalization properties of DQN**. 
- DQNs with **more aggressive generalization (larger off-diagonal terms in $K_{\theta}$)** are **less likely to demonstrate stable learning**.

---
# Reference

[Washington University - Line Search Methods](https://sites.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf)