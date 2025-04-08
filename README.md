# Reconstructing flow fields from sparse data using diffusion models

This repository contains the code used in my HiWi position (luis.medrano@tum.de) at the Physics-Based Deep Learning group at TUM. It is an extension of Marc Amoros MSc Thesis (marc.amoros@tum.de). The project was inspired by the work of [Shu et al.](https://arxiv.org/abs/2211.14680), where they perform super-resolution to flow-fields using diffusion models. We used the same network structure and dataset for comparing purposes. The main addition is a class to help benchmark the models and compare it with some baselines.

## Dataset
The used dataset can be downloaded on the [repo](https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution) of Shu et al. or through this [link](https://figshare.com/ndownloader/files/39181919). Then, it should be placed under the `data` folder, such that it can be found by the `KolmogorovFlowDataset` class from `datasets.py`. This class organizes the dataset in samples of 3 contiguous timesteps, which are used to train the models. There is also a second dataset added to the repository. Besides, in order to train the baselines on a classical supervised way with regression, we need to create a set of low resolution images, and a set of high resolution images. They can be created deleting a comment in the trainers and these datasets will be stored in new folders.

## Model
The architecture of the U-net network behind the model is the exact same as the one used by Shu et al., and the implementation found in `networks.py` is nearly identical to theirs. We fully reimplemented  the all the diffusion related parts of the model in the `diffusion.py` file, and the proposed masking procedure is implemented with the the method `ddim_mask()` of the `Diffusion` class.

## Training
The training of these models was done using the `trainer.py` script. Run `python trainer.py --help` to visualize all the possible parameters, or check one of the `tasks.sh` scripts for examples.

## Sampling examples
Examples of how to run the code to use the models can be found in the `experiments.py` script, where every function is an indepentend experiment, which loads the dataset, loads a pre-trained model, and generates samples using the diffusion model. 

## Use 

In `inference.py` you can find a general class to use the models and run some experiments. It also allows to compare the different variants of diffusion super-resolution models with its UNet baseline trained with regression, POD and the SwinIR model.