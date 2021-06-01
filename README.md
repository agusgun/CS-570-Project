# Exploratory Repository to Understand Group Normalization and Improving the Layer

## Important
- All of the VGG16 written in the notebook is VGG11. We are very sorry for these writing mistakes.
- All of the VGG plotted in the notebook is also VGG11.

## Understanding Part - Analysis Framework - [How Does Batch Normalization Help Optimization](https://arxiv.org/pdf/1805.11604.pdf)
1. Does a time-varying perturbation affect the training of GN?
2. The impact of GN layer on the training process of neural network
3. The impact of regularization on the training process of neural network with GN

## Improvement Part
1. Propose a new normalization layer by incorporating the underlying mechanism of BN to GN layer

## Guide - Improvement Experiment
1. Go to the `main_training_script` directory
2. Run `python assert_script.py` to download all of the needed datasets and check
3. Run `python main.py` with additional parameters based on your choice

## Reference
1. https://github.com/AlexeyGB/batch-norm-helps-optimization
2. https://github.com/kuangliu/pytorch-cifar
3. https://github.com/bearpaw/pytorch-classification