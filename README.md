# Hyperbolic Restricted Boltzmann Machine Neural Quantum State

This is the repo for the work arXiv: 2609.26032 [quant-ph] - Hyperbolic Restricted Boltzmann Machine NQS [https://arxiv.org/abs/2609.26032].

This is still under construction. 

- `trained_models`: contains the mpack and log files of trained RBM/HRBM NQS models for QSK system size $N=14$ to $N=20$.
- `trained_models_ii`: contains the mpack and log files of trained RBM/HRBM NQS models for QSK system size $N=22$ and $N=24$.
- `example_notebooks`: contains the jupyter notebooks for the training and inference of all RBM/HRBM NQS in two subfolders `training` and `inference`. Subfolder `csv_results` contains the processed results obtained by performing inference using the trained weights of the RBM/HRBM networks (the `mpack` files in the `trained_models` & `trained_models_ii` folders).
