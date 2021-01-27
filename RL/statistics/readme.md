# Statistics

Everything horrible about mathmatical satistics that is required to understand DRL.

---

## Information Entropy

### Shannon Entropy

For discrete random variable X and probability mass function P(X), we defien the Shannon Entropy **H(X)** as 

**H(X) = E[I(X)] = E[-log(P(X))]**

![](img/entropy/dist_entropy_sum.svg)

### Conditional Entropy

The conditional entropy of Y given X is defined as

![](img/entropy/cond_entropy.svg)
### Joint Entropy

The joint Shannon entropy (in bits) of two discrete random variables X and Y is defined as

![](img/entropy/joint_entropy.svg)

### Mutual Information(MI)

In general, MI is the intersection or say the common information for both random variables X and Y.

Let (X,Y) be a pair of random variables with values over the space X x Y. If their joint distribution is P(X,Y) and the marginal distributions are PX and PY, the mutual information is defined as

![](img/entropy/mutual_entropy_prod.svg)

In terms of summation, for discrete random variables X and Y

![](img/entropy/mutual_entropy_sum.svg)

In terms of integration, for continuous random variables X and Y

![](img/entropy/mutual_entropy_int.svg)

The relation between mutual information and conditional entropy and joint entropy

![](img/entropy/mutual_entropy_rel.svg)

With Venn diagram

![](img/entropy/mutual_entropy_venn.png)

For more detail, please refer to [this](https://en.wikipedia.org/wiki/Mutual_information)

## Jensen's Inequality

It states that for the convex transformation f(), the mean f(w * x + (1 - w) * y) of x, y on convex transform f is less than or equal to the mean applied after convex transformation w * f(x) + (1 - w) * f(y).

![](img/jensen/jensen.png)

Formaly, corresponding to the notation of the above figure, the Jensen's inequality can be defined as

![](img/jensen/jensen_formular.svg)

In probability theory, for a random variable X and a convex function φ, we can state the inequality as 

![](img/jensen/jensen_prob.svg)

## Variational Lower Bound

### Problem Setup

It is also knowd as **Evidence Lower Bound(ELBO)** or **VLB**. We can  assume that X are observations (data) and Z are hidden/latent variables. In general, we can also imagine Z as a parameter and the relationship between Z and X are represented as the following

![](./img/elbo/elbo_graph.png)

In the mean time, by the definition of Bayes' Theorem and conditional probability, we can get

![](./img/elbo/elbo_bayes.png)

### Proof

By the above statement, we can derive

![](./img/elbo/elbo_derive.png)


The equation(5) is the ELBO.

Where q(Z) in equation(2) is the approximattion the true posterior distribution p(Z|X), since we don't know the distribution of the p(Z|X) of hidden state Z. We also apply Jensen's inequality in equation(4).

Then, we denote L as ELBO as following

![](./img/elbo/elbo_L.png)

Since q(Z) is an qpproximation of p(Z|X), the KL-Divergence between q(Z) and p(Z|X) is >= 0. We can further derive:

![](./img/elbo/elbo_kl.png)

Then we rearrange the equation

![](./img/elbo/elbo_re.png)

### The application of ELBO

We can maximize the ELBO and minimize the KL-Divergence to maximize the logp(X). With KKT, we can rewrite the ***max(logp(X))*** to ***max(logp(X) - β KL[q(Z)||p(Z|X)])***

For more detail, please refer to this [handout](understanding-variational-lower.pdf) and [wikipedia](https://en.wikipedia.org/wiki/Evidence_lower_bound). They've given a great explaination. If you understand Chinese, you can read this [blog](https://odie2630463.github.io/2018/08/21/vi-1/). The author gives a series of blogs discussing about it.

## Information Bottleneck

[PDF Highlight](./The%20information%20bottleneck%20method.pdf)
## Deep Variational Information Bottleneck

[PDF Highlight](./deep_variational_information_bottleneck.pdf)

If you are a Chinese speaker, I recommand to read this [blog](https://zhuanlan.zhihu.com/p/102925788).



# Reference
- [漫談 Variational Inference (一)](https://odie2630463.github.io/2018/08/21/vi-1/)
- [Information Bottleneck](https://zhuanlan.zhihu.com/p/102925788)
- [《Auto-encoding Variational Bayes》阅读笔记](https://zhuanlan.zhihu.com/p/37224492)
- [Neural networks [7.8] : Deep learning - variational bound](https://www.youtube.com/watch?v=pStDscJh2Wo)
- [Wikipedia 互信息](https://zh.wikipedia.org/wiki/%E4%BA%92%E4%BF%A1%E6%81%AF)
- [Wikipedia 聯合分布](https://zh.wikipedia.org/wiki/%E8%81%94%E5%90%88%E5%88%86%E5%B8%83)
- [Wikipedia 熵 (信息論)](https://zh.wikipedia.org/wiki/%E7%86%B5_(%E4%BF%A1%E6%81%AF%E8%AE%BA))