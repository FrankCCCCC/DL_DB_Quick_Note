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

### Define A Neural Network

First, given a training dataset $\mathcal{D}$ including $N = |\mathcal{D}|$ data points. We denote the data point as $d_i = (x, y) \in \mathcal{D} \ \forall i$ where the feature vector $x \in \mathbb{R}^{k}$ and the label $y \in \mathbb{R}$. The set of the feature vectors is $\mathcal{X} = \{x: (x, y) \in \mathcal{D}\}$ and similarly, the set of the label $\mathcal{Y} = \{ y: (x, y) \in \mathcal{D}\}$. 

We represent the element of the neural network $f(x, \theta), \ x \in \mathcal{X}$ as following

$$
h^{1} = x W^{1} + b^{1}
\newline
h^{l+1} = a^l W^{l+1} + b^{l+1}
\newline
a^{l+1} = \phi(h^{h+1})
\newline
\hat{y} = f(x, \theta) = a^{L+1}
$$

where $\phi$ is the activation function, $h^{l+1}$ and $a^{l+1}$ are the pre-activation and the post-activation respectively. The parameter $\theta$ includes $W^{l}, b^{l} \in \theta \ \forall l$. $W^{l+1} \in \mathbb{R}^{n_l \times n_{l+1}}$ and $b^{l+1} \in \mathbb{R}^{1 \times n_{l+1}}$ are the weight and the bias respectively. They follow the LeCun and normal distribution respectively.

$$
W_{i,j}^l = \frac{\sigma_w}{\sqrt{n_l}} w_{i,j}^l, \quad b_{j}^l = \sigma_b \beta_{j}^l 
\newline
w_{i,j}^l, \beta_{j}^l \overset{i.i.d}{\sim} \mathcal{N}(0, 1)
$$

Let $\theta^{l}$ denote as the all parameters of the layer $l$.

$$
\theta^{l} = vec({W^l, b^l}) \in \mathbb{R}^{(n_{l} + 1) \times n_{l-1}}
\newline
\theta = vec(\cup_{l=1}^{L+1} \theta^l)
$$

We also denote the empirical loss of all training dataset as $\mathcal{L}(\mathcal{X}, \mathcal{Y})$, and $l(\hat{y}, y)$ as loss function. 

$$
l(\hat{y}, y) = l(f(x, \theta), y)
\newline
\mathcal{L}(\mathcal{X}, \mathcal{Y}) = \sum_{x \in \mathcal{X}, \ y \in \mathcal{Y}} l(f(x, \theta), y)
$$

## Infinite-Width Neural Network As A Linear Model

### Linear Model

![](img/w_changing_training.png)

As we've shown in the previous post, the parameters of the neural network change more slightly while the width of the network gets larger. In the other words, the neural network **remains almost unchanged during training.** As a result, the parameters $\theta^{(T)}$ of neural network after training $T$ steps will be very close to the initial parameters $\theta^{(0)}$.

### Taylor Expansion

Since the parameters of the infinite-width neural network only change slightly, thus, we can expand the neural network with Taylor expansion.

Taylor expansion

$$
g(x) = \sum_{n=0}^{\infty} \frac{g^{(n)}(a)}{n!} (x - a)^n
$$

First-order Taylor expansion

$$
g(x) \approx \ g(a) + \frac{d g(a)}{dx} (x - a)
$$

We denote the parameters of the neural network at training step $t$ as $\theta^{(t)}$. Then, expand the neural network $f(x, \theta^{(t)})$ at training step $t$ with data point $x$.

$$
\hat{y}^{(t)} = f(x, \theta^{(t)}) \approx \bar{f}(x, \theta^{(t)}) = f(x, \theta^{(0)}) + \nabla_{\theta} f(x, \theta^{(0)})(\theta^{(t)} - \theta^{(0)}), \ \forall t
$$

where $\hat{y}^{(t)}$ is the prediction of the data point $x$ from the network at training step $t$

As for the whole dataset $\mathcal{X}$, the predictions $\hat{\mathcal{Y}}^{(t)}$

$$
\hat{\mathcal{Y}}^{(t)} = f(\mathcal{X}, \theta^{(t)}) \approx \bar{f}(\mathcal{X}, \theta^{(t)}) = f(\mathcal{X}, \theta^{(0)}) + \nabla_{\theta} f(\mathcal{X}, \theta^{(0)})(\theta^{(t)} - \theta^{(0)}), \ \forall t
$$

### Combine With Gradient Descent

The gradient descent

$$
\theta^{(t)} = \theta^{(0)} + \eta \nabla_{\theta} \mathcal{L}(\mathcal{X}, \mathcal{Y}) 
= \theta^{(0)} + \eta \nabla_{\theta} f(\mathcal{X}, \theta^{(0)}) \nabla_{f(\mathcal{X}, \theta^{(0)})} \mathcal{L}(\mathcal{X}, \mathcal{Y})
$$

$$
\theta^{(t)} - \theta^{(0)} = \eta \nabla_{\theta} f(\mathcal{X}, \theta^{(0)}) \nabla_{f(\mathcal{X}, \theta^{(0)})} \mathcal{L}(\mathcal{X}, \mathcal{Y})
$$

Replace

$$
\bar{f}(\mathcal{X}, \theta^{(t)}) = f(\mathcal{X}, \theta^{(0)}) + \eta \nabla_{\theta} f(\mathcal{X}, \theta^{(0)})^{\top} \nabla_{\theta} f(\mathcal{X}, \theta^{(0)}) \nabla_{f(\mathcal{X}, \theta^{(0)})} \mathcal{L}(\mathcal{X}, \mathcal{Y})
$$

Let $T_{\mathcal{X} \mathcal{X}}^{(0)} = \nabla_{\theta} f(\mathcal{X}, \theta^{(0)})^{\top} \nabla_{\theta} f(\mathcal{X}, \theta^{(0)})$

$$
= f(\mathcal{X}, \theta^{(0)}) + \eta T_{\mathcal{X} \mathcal{X}}^{(0)} \mathcal{L}(\mathcal{X}, \mathcal{Y})
$$

where $T^{(0)}_{\mathcal{X} \mathcal{X}} \in \mathbb{R}^{|\mathcal{D}| \times |\mathcal{D}|}$ is the **Neural Tangent Kernel(NTK)**

## Gradient Flow Of MSE As A Linear Regression

In the previous section, we've derive the relation between the prediction of the neural network and the gradient descent at time step $t$. In this section, we'll dive into the point of view of gradient flow. For now, we've known that 

$$
\bar{f}(\mathcal{X}, \theta^{(t)}) = f(\mathcal{X}, \theta^{(0)}) + \eta T_{\mathcal{X} \mathcal{X}}^{(0)} \mathcal{L}(\mathcal{X}, \mathcal{Y})
$$

