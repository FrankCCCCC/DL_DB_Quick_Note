---
marp: true
theme: default
---

# Towards Characterizing Divergence in Deep Q-Learning

Citation: 40

OpenAI

*Joshua Achiam, Ethan Knight, Pieter Abbeel*

---

# Problem Setup
Why DQN cannot converge?
How about analyzing DQN with NTK?

---
# Main Idea & Conclusion

- Analyze DQN with NTK
- The main reason why DQN diverge(non-converge) is Over-generalization.
- Propose Pre-DQN to 
- Pre-QN

---
# Outline

- Main Ideas & Conclusion
- Analyzation Setup
- Building Intuition for Divergen with NTK
- Pre-QN
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