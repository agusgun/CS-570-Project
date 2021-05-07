# Exploratory Repository to Understand Group Normalization and Improving the Layer

## Understanding Part
1. Does a time-varying perturbation affect the training or testing result of GN? Refer to [How Does Batch Normalization Help Optimization](https://arxiv.org/pdf/1805.11604.pdf) NIPS 2018 paper
2. How about if we group the channel randomly or alternately? What effect will this perturbation give?

Observe the perturbation effect based on the gradient predictiveness, training accuracy, training loss, and testing loss and accuracy
Another idea?

## Improvement Part



## To do:
<p>Add more dataloaders.

<p>Implemented the metrics which can be used to compare the distribution difference.
Test
Try Again

## Reference
1. https://github.com/AlexeyGB/batch-norm-helps-optimization
2. https://github.com/kuangliu/pytorch-cifar