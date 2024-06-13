# GPU-accelerated GCMC calculations

This repository contains Python scripts that implement GCMC simulations capable of running on GPUs. It supports various machine learning force fields (MLFF) as well as classical force fields. For MLFFs, we utilize pre-trained graph neural network models trained on the ODAC23 dataset. Please refer to the ODAC23 [paper](https://pubs.acs.org/doi/10.1021/acscentsci.3c01629) and the [website](https://open-dac.github.io/) for more details about the dataset.

## Available force fields
- Classical force fields
  - Lennard-Jones + electrostatics
- [MLFF](https://fair-chem.github.io/core/model_checkpoints.html#s2ef-models)
  - DimeNet++
  - SchNet
  - PaiNN
  - GemNet-OC
  - eSCN
  - EquiformerV2

## Dependencies
- NumPy
- PyTorch
- [ocp](https://github.com/Open-Catalyst-Project)
