---
marp: true
theme: default
paginate: true
# _class: invert
# color: white
# backgroundColor: black
class: lead
---

# Neural Kernel Without Tangents

---

# Motivation

- NTK, CNTK... do not match the performance of neural networks on most tasks of interest.
- The NTK constructions themselves are not only hard to compute, but their mathematical formulae are difficult to even write down.

---

# Problem Formulation

- Are there computationally tractable/easier kernels that approach the expressive power of neural networks?
- Is there a correlation between neural architecture performance and the performance of the associated kernel?

---

# Outline

- Main Idea
- Experiments
- Conclusion

---

# Main Idea

- Construct CNN architecture using only **$3 \times 3$ convolutions, $2 \times 2$ average pooling, ReLU**.
- **Compositional Kernel**: Kernelize $1..., L$ layers as kernel functions $k_{1}..., k_{L}$ and **compute the kernel hierarchily $k_{L}(k_{L-1}(...k_{1}(x, y)))$** as the kernel of the corresponding CNN architecture.
- **5-layers compositional kernel**(in Myrtle5 architecture) can **significantly outperform(about 10% classification accuracy)** than **14-layers CNTK on CIFAR-10([Arora et al. 2020](https://iclr.cc/virtual_2020/poster_rkl8sJBYvH.html))** while the **training samples are less than 1000**.

![](img/myrtle5.png)

---

- **Bag of features** is simply a generalization of a matrix or tensor: whereas a matrix is an indexed list of vectors, a bag of features is a collection of elements in a Hilbert space $\mathcal{H}$ with a finite, structured index set $\mathcal{B}$. 
- EX: we can consider an image to be a bag of features where the index set $\mathcal{B}$ is the pixel’s row and column location and $\mathcal{H}$ is $\mathbb{R}^3$: at every pixel location, there is a corresponding vector in $\mathbb{R}^3$.
- 

---

# Convolution Kernel

![width:800px](img/convolution.png)

---

# Average Pooling Kernel

![width:800px](img/avg_pooling.png)

---

# ReLU Kernel

![width:800px](img/relu1.png)

---

# ReLU Kernel

![width:800px](img/relu2.png)

---

