# Proof of Gaussian Process

For $X \in \mathbb{R}^{n}$

$$
X \sim \mathcal{N}(\mu, \Sigma)
$$

$$
p_X(x; \mu, \Sigma) = p(X = x) = \frac{1}{det(\Sigma) \sqrt{(2 \pi)^n}} e^{-\frac{1}{2} (x - \mu)^{\top} \Sigma^{-1} (x - \mu)}
$$

## Conditional Distribution

Given a joint probability distribution

$$
X \sim \mathcal{N}(\mu, K)
$$

$$
X
= \left[
    \begin{matrix}
        X_1 \\
        X_2
    \end{matrix}
\right], \quad
\mu 
= \left[
    \begin{matrix}
        \mu_1 \\
        \mu_2
    \end{matrix}
\right], \quad
K
= \left[
    \begin{matrix}
        K_{11} & K_{12} \\
        K_{21} & K_{22}
    \end{matrix}
\right]
$$

$$
p(X_2 | X_1) = \frac{p(X_2, X_1)}{\int p(X_2, X_1) dX_2} =  \frac{p(X_2, X_1)}{p(X_1)}
$$

$$
\frac{}{}
$$

---

$$
Var[X_1 | X_2] = Var[Z] = Var[X_1 + A X_2]
$$

$$
= Var[X_1] + Var[A X_2] + Cov[X_1, A X_2] + Cov[A X_2, X_1]
$$

$$
= Var[X_1] + A Var[X_2] A^{\top} + Cov[X_1, X_2] A^{\top} + A Cov[X_2, X_1]
$$

$$

$$

Variance of Sum of Random Variables

$$
Var[X_1 + X_2] = E[((X_1 + X_2) - (E[X_1] + E[X_2]))((X_1 + X_2) - (E[X_1] + E[X_2]))^{\top}]
$$

$$
= E[((X_1 - E[X_1]) + (X_2 - E[X_2])) ((X_1 - E[X_1]) + (X_2 - E[X_2]))^{\top}]
$$

$$
= E[(X_1 - E[X_1]) (X_1 - E[X_1])^{\top}] 
+ E[(X_2 - E[X_2]) (X_2 - E[X_2])^{\top}] 
+ E[(X_1 - E[X_1]) (X_2 - E[X_2])^{\top}] 
+ E[(X_2 - E[X_2]) (X_1 - E[X_1])^{\top}]
$$

$$
= Var[X_1] + Var[X_2] + Cov[X_1, X_2] + Cov[X_2, X_1]
$$

Variance of Random Vector After Linear Transform

$$
Var[A X + B] = E[(A X + B - E[A X + B]) (A X + B - E[A X + B])^{\top}]
$$

$$
= E[(A (X - E[X])) (A (X - E[X]))^{\top}]
$$

$$
= E[(A (X - E[X])) ((X - E[X])^{\top} A^{\top})]
$$

$$
= A E[(X - E[X])) ((X - E[X])^{\top}] A^{\top}
$$

$$
= A Var[X] A^{\top}
$$

Covariance of  Random Vector

$$
Cov[X_1, A X_2 + B] = E[(X_1 - E[X_1]) (A X_2 + B- E[A X_2 + B])^{\top}]
$$

$$
= E[(X_1 - E[X_1]) (X_2 - E[X_2])^{\top} A^{\top}]
$$

$$
= Cov[X_1, X_2] A^{\top}
$$

$$
Cov[AX_1, X_2] = E[(A X_1 + B- E[A X_1 + B]) (X_2 - E[X_2])^{\top}]
$$

$$
= E[A (X_1 - E[X_1]) (X_2 - E[X_2])^{\top}]
$$

$$
= A Cov[X_1, X_2]
$$