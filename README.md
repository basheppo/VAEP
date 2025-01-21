# Variational Autoencoder (VAE) for Multiscale Simulation of Spatially Correlated Microstructures

This repository contains an implementation of a Variational Autoencoder (VAE) inspired by the paper **"Multiscale Simulation of Spatially Correlated Microstructure via a Latent Space Representation"**. The code is implemented in TensorFlow and is designed for the generation and prediction of spatially correlated microstructures.

---

## Overview

This VAE implementation serves as a generative model for high-dimensional spatially correlated microstructures. It employs:
- **Encoder**: Compresses the input into a low-dimensional latent space.
- **Decoder**: Reconstructs the input from the latent space representation.
- **Predictor**: Predicts material properties from the latent space.

The model supports both training and testing workflows, with built-in loss functions for reconstruction, KL divergence, and property prediction.

---

## Features

- **Customizable Architecture**: Adjustable latent dimension, input shapes, and model parameters.
- **Predictor Component**: Simultaneous prediction of material properties using the latent representation.
- **Modular Design**: Separate methods for building encoder, decoder, and predictor models.
- **Efficient Training**: Gradient-based optimization using TensorFlow's GradientTape.
- **Comprehensive Loss Metrics**:
  - Reconstruction Loss
  - KL Divergence
  - Prediction Loss

---

## Model Architecture

### Encoder
- A series of 3D convolutional layers followed by a dense layer.
- Outputs the latent mean, variance, and sampled latent vector.
  
### Decoder
- A series of transposed convolutional layers to reconstruct the input from the latent space.

### Predictor
- Fully connected layers for predicting properties from the latent representation.

### Sampling Layer
- Implements the reparameterization trick to sample latent vectors from a Gaussian distribution.

---

## Installing Dependencies

To set up the environment, install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Citation

This is the source of the original paper:  

```bibtex
@article{JONES2024112966,
  title = {Multiscale simulation of spatially correlated microstructure via a latent space representation},
  journal = {International Journal of Solids and Structures},
  volume = {301},
  pages = {112966},
  year = {2024},
  issn = {0020-7683},
  doi = {https://doi.org/10.1016/j.ijsolstr.2024.112966},
  url = {https://www.sciencedirect.com/science/article/pii/S0020768324003251},
  author = {Reese E. Jones and Craig M. Hamel and Dan Bolintineanu and Kyle Johnson and Robert {Buarque de Macedo} and Jan Fuhg and Nikolaos Bouklas and Sharlotte Kramer},
  keywords = {Multiscale simulation, Elastoplasticity, Variational autoencoder, Structure–property map, Latent space, Finite size effects, Spatial correlation, Functional gradation},
  abstract = {When deformation gradients act on the scale of the microstructure of a part due to geometry and loading, spatial correlations and finite-size effects in simulation cells cannot be neglected. We propose a multiscale method that accounts for these effects using a variational autoencoder to encode the structure–property map of the stochastic volume elements making up the statistical description of the part. In this paradigm the autoencoder can be used to directly encode the microstructure or, alternatively, its latent space can be sampled to provide likely realizations. We demonstrate the method on three examples using the common additively manufactured material AlSi10Mg in: (a) a comparison with direct numerical simulation of the part microstructure, (b) a push forward of microstructural uncertainty to performance quantities of interest, and (c) a simulation of functional gradation of a part with stochastic microstructure.}
}
```
