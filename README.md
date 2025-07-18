# GraphCMC

Welcome to the GraphCMC repository! This package is designed to advance the application of machine learning in molecular simulations through the integration of traditional and advanced machine learning force fields. Our scripts enhance GCMC simulations, providing the community with the tools needed to conduct detailed studies with both classical and MLFFs.

## ML-Enhanced GCMC Simulations
This repository houses Python scripts that enable GCMC simulations integrating a variety of force fields optimized for accuracy and efficiency. Our focus is on the incorporation of pre-trained MLFFs, utilizing state-of-the-art graph neural network models trained on the expansive ODAC23 dataset. For further details on the dataset and ML models, please refer to the ODAC23 [paper](https://pubs.acs.org/doi/10.1021/acscentsci.3c01629) and the [website](https://open-dac.github.io/).

## Dependencies
- [fairchem-core==0.10.0](https://pypi.org/project/fairchem-core/1.10.0/)
- `torch-scatter`
- `torch-sparse`

> **Note**: `graphcmc` currently depends on `fairchem-core==0.10.0` and does not yet support models introduced in `fairchem-core v2`. Support for newer versions is planned. Stay tuned for updates!

## Supported Force Fields
### Classical Force Fields (LJ + electrostatics)
- UFF

### ML Force Fields
The following GNN-based MLFFs trained on the ODAC23 dataset are supported via `fairchem.core.OCPCalculator`. You can find the downloadable model checkpoints [here](https://fair-chem.github.io/dac/models.html).

- SchNet-S2EF-ODAC
- DimeNet++-S2EF-ODAC
- PaiNN-S2EF-ODAC
- GemNet-OC-S2EF-ODAC
- eSCN-S2EF-ODAC
- EquiformerV2-S2EF-ODAC
- EquiformerV2-Large-S2EF-ODAC

## Setting Up Your Development Environment

To ensure reproducibility and consistency, we recommend using a Conda environment with Python 3.9.

### 1. Clone the repository

```bash
git clone git@github.com:sihoonchoi/graphcmc.git
cd graphcmc
```

### 2. Install the package

Run the following command to install `graphcmc` with extra dependencies:

```bash
pip install -e .[torch-extensions]
```

## Running GCMC Calculations

Please refer to the [tutorial]() for detailed instructions on running GCMC calculations using `graphcmc`.

## Acknowledgements

- The Ewald summation implementation is adapted from [VaspBandUnfolding](https://github.com/QijingZheng/VaspBandUnfolding).

- The Peng-Robinson Equation of State part and associated critical property data are adapted from the work of [Goeminne *et al.*](https://doi.org/10.1021/acs.jctc.3c00495).
