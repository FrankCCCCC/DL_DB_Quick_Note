# Lottery Hypothesis & Model Compression

### Can we get a better and compact model?

---

### Proving the Lottery Ticket Hypothesis Pruning is All You Need

[Notes](./Proving_the_Lottery_Ticket_Hypothesis_Pruning_is_All_You_Need.pdf)
  
Try to prove Lottery Hypothesis with math

---

### The Lottery Ticket Hypothesis Finding Sparse Trainable Neural Networks

[Notes](./THE_LOTTERY_TICKET_HYPOTHESIS_FINDING_SPARSE_TRAINABLE_NEURAL_NETWORKS.pdf)

#### Introduction

提出Lottery Hypothesis，在一個神經網路中，存在一個較小的網路(called Lottery Ticket or Winning Ticket, 10-20% or less of the original network size)，且以原網路的方式Initialize後並Retrain可以得到比原網路同等或是更好的Test Accuracy。然而，若用Random Initialize的並訓練，則學習速度和Test Accuracy都會下降。

#### Theorem: Lottery Hypothesis

The lottery ticket hypothesis predicts that ∃ m for which j
0 ≤ j (commensurate training time), a 0 ≥ a (commensurate accuracy), and kmk0  |θ| (fewer parameters)

給定任一神經網路θ，Exist 一個神經網路θ'，及Mask m = {0, 1}，使 = m * θ，θ >> θ'，分別訓練神經網路θ' j' iteration、θ j iteration，在同量的(commensurate)且訓練量下j' <= j，神經網路θ, θ'可以達到Test Accuracy a, a'，且a' >= a

> commensurate: 同量的，相稱的

#### Identifying winning tickets

**Steps:**

1. Randomly initialize a neural network f(x; θ_0) (where θ_0 ∼ D_θ).
   
2. Train the network for j iterations, arriving at parameters θ_j .
   
3. Prune p% of the parameters in θ_j , creating a mask m.
   
4. Reset the remaining parameters to their values in θ0, creating the winning ticket f(x; m * θ_0).

> 每次Train神經網路j個iteration後，prune p%的神經元，如此反覆訓練、修剪(Iterative Pruning)for n round，就可以得到Lottery Tickets

#### Winning Tickets in Fully-Connected Networks

**Iterative Pruning**

#### Winning Tickets in Convolutional Networks

#### VGG and ResNet for CIFAR10

#### Result

1. When randomly reinitialized, a winning ticket
learns more slowly and achieves lower test accuracy, suggesting that initialization is important to
its success

> Random Init會使的Winning Ticket學習速度較慢且得到較差的Test Accuracy，因此Initialization是相當重要的

2. Usually, the winning tickets we find are 10-20% (or less) of the size of the original network
 
---

### Rethinking The Value of Network Pruning

[Note](./RETHINKING_THE_VALUE_OF_NETWORK_PRUNING.pdf)

#### Introduction

**針對Pruning 提出三個結論**

1. training a large, over-parameterized
model is often not necessary to obtain an efficient final model

> (針對3-Stage Network Pruning Pipeline)訓練一個過參數化(Over-parameterized)的模型，對於找到最終小而有效的模型並不必要

2. learned “important” weights of the large model are typically not useful for the small pruned
model

> 對於小模型來說，學習大模型訓練出來重要的Weight並沒有幫助，也就是說，傳統Pruning繼承大Model再Prune最後Fine-Tune的做法沒有必要

3. the pruned architecture itself, rather than a set of inherited “important”
weights, is more crucial to the efficiency in the final model

> Pruned的Architecture比Weight重要

**兩個出人預料的觀察**

1. For structured pruning methods with
predefined target network architectures, directly training the small target model from random initialization can achieve the same

> 使用Structured Pruning(Predefine 好Pruning Rate)後得到的Model，Random Initialize後訓練可以達到和原本大Model一樣(甚至更好)的Performance

2. For structured pruning methods with autodiscovered target networks, training the pruned model from scratch can also achieve comparable or even better performance than fine-tuning. This observation shows that for these pruning methods, what matters more may be the obtained architecture, instead of the preserved weights

> Pruning後得到真正重要的資訊是Architecture，而非Weights

**小結**

Pruning算法最後得到的小模型其實重點在於Pruning完後的"Network Architecture"，大模型訓練出的Weight其實並不重要，在得到小模型後直接Random Init在訓練也可以得到相同甚至更好的Test Accuracy。

#### Sructured Pruning & Unstructured Pruning

- Sructured Pruning:
  Predefine p% pruning rate(pruned nodes / all nodes) for each layers

- Unstructured Pruning:
  Prune with different pruning rate(pruned nodes / all nodes) for each layers automatically

![Pruning methods](imgs/rethink/pruning_methods.png)
*Sructured Pruning & Unstructured Pruning*

#### EXPERIMENTS ON THE LOTTERY TICKET HYPOTHESIS 

In this section, the authors do some experiments and get different results that "random initialization is enough for the pruned model to achieve competitive performance".

幾個重要的且和原Lottery Hypothesis原作不一樣的evaluation settings:

1. 比較了Strucred Pruning的結果，而在Lottery Hypothesis的Paper中，作者只有比較Unstrucred Pruning的結果

2. 使用較Modern且較大的NN，Lottery Hypothesis的原作者使用較淺的NN(Layer數 < 6)
   
3. 使用Momentum SGD和較大的Initial LR(Learning Rate = 0.1, 常用在Image Classification)
   
4. 使用較大的Dataset(ImageNet Dataset)，Lottery Hypothesis原作只用MNIST和CIFAR


**Result:**

1. 在非結構化剪枝(Unstructured Pruning)中，LR(Learning Rate)較大的時候(0.1)，Lottery Ticket相比於Random Init沒有太多優勢。如果LR(Learning Rate)較小(0.1)，結果誠如Lottery Hypothesis所言，Lottery Ticket 確實比Random Init好。但是在小LR的狀況下，無論是Lottery Ticket和Random Init結果都比大LR的結果差

![Compare to lottery ticket in unstructured pruning1](imgs/rethink/unstructured_prune_res.png)
*LR(Learning Rate)較大的時候(0.1)，Lottery Ticket相比於Random Init沒有太多優勢；反之如果LR(Learning Rate)較小(0.1)，Lottery Ticket 確實比Random Init好*

![Compare to lottery ticket in unstructured pruning1](imgs/rethink/unstructured_prune_res2.png)
*小LR使Lottery Ticket比大LR的兩者(Lottery Ticket, Random Init)都更差*
   
2. 而在結構化剪枝(Structured Pruning)中，Lottery Ticket並不會帶來比Random Init更好的Test Accuracy

![Compare to lottery ticket in structured pruning2](imgs/rethink/structured_prune_res.png)


引述Paper結論

To summarize, in our evaluated settings, the winning ticket only brings improvement in the case of unstructured pruning, with small initial learning rate, but this small learning rate yields inferior accuracy compared with the widely-used large learning rate.

---

### Dawing Early-Bird Tickets More Efficient Training of Deep Networks

[Note](./DRAWING_EARLY-BIRD_TICKETS_TOWARDS_MORE_EFFICIENT_TRAINING_OF_DEEP_NETWORKS.pdf)

#### Introduction

Discover the Early-Bird (EB) tickets phenomenon: the winning tickets can be drawn very early in training(6.25%, 12.5% in experiments), and with aggressively low-cost training algorithms(5.8 ~ 10.7* energy saving)

#### Early-Bird (EB) Tickets Hyppothesis

Consider a dense, randomly-initialized network f(x; θ), f reaches a minimum validation loss floss at the i-th iteration with a test accuracy facc, when optimized with SGD on a training set. 

In addition, consider subnetworks f(x; m ⊙ θ) with a mask m ∈ {0, 1} indicates the pruned and unpruned connections in f(x; θ). When being optimized with SGD on the same training set, f(x; m ⊙ θ) reach a minimum validation loss f′ loss at the i′-th iteration with a test accuracy f′acc. 

The EB tickets hypothesis articulates that there **exists m
such that f′acc ≈ facc (even ≥), i.e., same or better generalization, with i′ ≪ i (e.g., early stopping)
and sparse m (i.e., much reduced parameters).**

> 存在一個sparse的mask m = {0, 1}，使得EB Ticket subnetwork m' = m ⊙ θ，可以在訓練到第i'-th iteration時，就達到f' accuracy，而i' << i 且 f' >= f，當原network θ 在第i-th iteration達到f accuracy

#### Hypothesis Validation

1. Do EB Ticket Always Exist?
   
   p is pruning rate. Sometime over pruning(70% on PreNet101) will make drawing ticket harder

   - EB tickets always emerge at very early stage

   - Some EB tickets are outperform than unpruned full-trained model

   ![ebExist](imgs/eb/ebExist.png)
   *EB Ticket training epochs & Retrain Accuracy*

2. Do EB Tickets Still Emerge under Low-Cost Training?
   
   The meaning of [80, 120] is starting from 0.1, decay to 0.01 at 80-th epoch and further decay to 0.001 at 120 epoch
   
   - Appropriate **Large Learning Rate** is important for emerging EB tickets. The EB tickets always emerge at larger learning rate whose retrain accuray is also better

    ![emergeLowCost](imgs/eb/lowCostLR.png)
    *Learning Rate Schedule & Retrain Accuracy*
  
   - **Low-Precision Training** Dost Not Destroy EB Tickets
     Train & prune the original model with only 8 bits(for all modle weights, activations, gradients and errors). The EB ticket still emerge at very early stage. Then they retrain the EB ticket with full precision. It aggressively save energy.

    ![lowPrecision](imgs/eb/lowPrecision.png)
    *Low-Precision Pruning & Retrain Accuracy*
   
3. How to Implement EB Tickets?

#### Experiments

#### Conclusion

### Learning Efficient Convolutional Networks through Network Slimming

[Note](./Learning_Efficient_Convolutional_Networks_through_Network_Slimming.pdf)

### Weight Agnostic Neural Networks

[Note](./Weight_Agnostic_Neural_Networks.pdf)
