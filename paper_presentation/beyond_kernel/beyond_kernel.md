---
marp: true
theme: default
paginate: true
# _class: invert
# color: white
# backgroundColor: black
class: lead
---

# What Can ResNet Learn Efficiently, Going Beyond Kernels?

NIPS'19, Citation: 77

Zeyuan Allen-Zhu, Yuanzhi Li

Microsoft Research AI, Carnegie Mellon University

---

# Motivation

- In many practical tasks, neural networks give much better generalization error compared to kernels, although both methods can achieve zero training error.
- For example, ResNet achieves 96% test accuracy on the CIFAR-10 data set, but NTKs achieve 77% and random feature kernels achieve 85%. This gap becomes larger on more complicated data sets.

---

# Problem Formulation

Can neural networks(like ResNet) efficiently and distribution-freely learn a concept class, with better generalization than kernel methods?

---

# Main Idea

- For neural networks with ReLU activations, we show without any distributional assumption, a **three-layer residual network (ResNet)** can (improperly) learn a **concept class that includes three-layer ResNets of smaller size and smooth activations**, and the generalization error is also small if polynomially many training examples are given while the network is trained by SGD.
- Then prove that for **some $\delta \in (0, 1)$, with $N = O(\delta^{-2})$ training samples**, neural networks can efficiently **achieve generalization error $\delta$ for this concept class over any distribution**; in contrast, there exists a simple distributions such that **any kernel method cannot have generalization error better than $\sqrt{\delta}$ for this class cannot have generalization error better than $\sqrt{\delta}$ for this class**.
- Also prove a computation complexity advantage of neural networks with respect to linear regression over arbitrary feature mappings as well.

