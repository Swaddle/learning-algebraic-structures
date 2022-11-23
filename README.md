# learning-algebraic-structures

Learning algebraic structures with machine learning.

Attempting to reproduce the results of the paper [[0](https://arxiv.org/abs/1905.02263)].


## Requirements

This repo uses `git-lfs`.

To generate the Cayley tables and Latin squares from [[0](https://arxiv.org/abs/1905.02263)] you will need `gap` and `sage`.
However, there is a serialised/pickled list of torch tensors for Cayley tables in `cayleys.p`, and a serialised list Latin squares (list of lists) in `latins.p` (be careful with pickled files, they are not secure in general, if you don't trust us, generate these yourself). 

Running the model you will need `torch`. 

## About

Motivated by the brief description of the network used in [[0](https://arxiv.org/abs/1905.02263)], we first try a very simple fully connected network with linear layers. Our first change though is using ReLu activations instead of softmax.

## References 

[0] [Learning Algebraic Structures: Preliminary Investigations](https://arxiv.org/abs/1905.02263)