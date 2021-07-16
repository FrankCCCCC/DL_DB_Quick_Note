# NTK Proof

$$
f(x, \theta) = \sigma(wx + b) \quad  w, b \in \theta
$$

$$
L(f, x, y, \theta) = (f(x, \theta) - y)^2
$$

$$
\theta_{t+1} = \theta_t + \eta \nabla_{w} L(f, x, y, \theta)
$$

Denote $\theta_t$ as dynamic $\theta(t)$

$$
\frac{d\theta(t)}{dt} = \nabla_{\theta} L(f, x, y, \theta)
$$

$$
= 2(f(x, \theta) - y) \nabla_{\theta} f(x,\theta)
$$

$$
= 2(f(x, \theta) - y) \nabla_{\theta} (f(x,\theta_0) + \nabla_{\theta} f(x,\theta_0)^{\top}(\theta - \theta_0))
$$

$$
= 2(f(x, \theta) - y) \nabla_{\theta} (f(x,\theta_0) + \nabla_{\theta} f(x,\theta_0)^{\top}\theta - \nabla_{\theta} f(x,\theta_0)^{\top}\theta_0)
$$

$$
= 2(f(x, \theta) - y) \nabla_{\theta} f(x,\theta_0)
$$

---

## Abstract

To understand what is the neural tangent kernel(NTK), there are 3 important result that need to remember. 

- When the width of the neural network goes to infinity, the network will be equivalent to a Gaussian process.

- When the width of the neural network goes to infinity, the weight of the network will remain  almost unchanged during training. That is, the neural network will become a linear model.

- Since the network will become a linear model, the loss surface of the MSE of the infinite-width network will be a convex. As a result, we can optimize the infinite-width network just like optimizing a linear regression and solve it by ODE.

Combine these 3 points, NTK is a kernel that can kernelize the neural network architecture and it provides a closed-form solution of the kernel for anytime. Thus, We can compute the NTK at the end of training without actual training and compute the posterior and the prediction of the testing data with Bayesian inference.

## Infinite-Width Neural Network As Gaussian Process



## Infinite-Width Neural Network As A Linear Model



## Gradient Flow Of MSE As A Linear Regression