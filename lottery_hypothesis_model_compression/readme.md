# Lottery Hypothesis & Model Compression
### Can we get a better and compact model?

### Proving the Lottery Ticket Hypothesis Pruning is All You Need
[Notes](./Proving%20the%20Lottery%20Ticket%20Hypothesis%20Pruning%20is%20All%20You%20Need.pdf)
  
  Try to prove Lottery Hypothesis with math

### The Lottery Ticket Hypothesis Finding Sparse Trainable Neural Networks

[Notes](./THE_LOTTERY_TICKET_HYPOTHESIS_FINDING_SPARSE_TRAINABLE_NEURAL_NETWORKS.pdf)

提出Lottery Hypothesis的原論文

**Theorem: Lottery Hypothesis**

The lottery ticket hypothesis predicts that ∃ m for which j
0 ≤ j (commensurate
training time), a
0 ≥ a (commensurate accuracy), and kmk0  |θ| (fewer parameters

給定任一神經網路θ，Exist 一個神經網路θ'，及Mask m = {0, 1}，使 = m * θ，θ >> θ'，分別訓練神經網路θ' j' iteration、θ j iteration，在同量的(commensurate)且訓練量下j' <= j，神經網路θ, θ'可以達到Test Accuracy a, a'，且a' >= a

commensurate: 同量的，相稱的

**Identifying winning tickets**

Steps:

1. Randomly initialize a neural network f(x; θ_0) (where θ_0 ∼ D_θ).
2. Train the network for j iterations, arriving at parameters θ_j .
3. Prune p% of the parameters in θ_j , creating a mask m.
4. Reset the remaining parameters to their values in θ0, creating the winning ticket f(x; m * θ_0).

每次Train神經網路j個iteration後，prune p%的神經元，如此反覆訓練、修剪(Repeated train & prune)for n round，就可以得到Lottery Tickets

**Result**

- Usually, the winning tickets we find are 10-20% (or less) of the size of the original network

### Rethinking The Value of Network Pruning

[Note](./RETHINKING%20THE%20VALUE%20OF%20NETWORK%20PRUNING.pdf)

### Dawing Early-Bird Tickets More Efficient Training of Deep Networks

[Note](./DRAWING_EARLY-BIRD_TICKETS_TOWARDS_MORE_EFFICIENT_TRAINING_OF_DEEP_NETWORKS.pdf)

### Weight Agnostic Neural Networks

[Note](./Weight_Agnostic_Neural_Networks.pdf)

### Learning Efficient Convolutional Networks through Network Slimming

[Note](./Learning_Efficient_Convolutional_Networks_through_Network_Slimming.pdf)