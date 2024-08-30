# GraphMC

Welcome to the GraphMC repository! This platform is designed to advance the application of machine learning in molecular simulations through the integration of traditional and advanced machine learning force fields. Our scripts enhance GCMC simulations, providing the community with the tools needed to conduct detailed studies with both classical and MLFFs.

## ML-Enhanced GCMC Simulations
This repository houses Python scripts that enable sophisticated GCMC simulations integrating a variety of force fields optimized for accuracy and efficiency. Our focus is on the incorporation of pre-trained MLFFs, utilizing state-of-the-art graph neural network models trained on the expansive ODAC23 dataset. For further details on the dataset and ML models, please refer to the ODAC23 [paper](https://pubs.acs.org/doi/10.1021/acscentsci.3c01629) and the [website](https://open-dac.github.io/).

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
