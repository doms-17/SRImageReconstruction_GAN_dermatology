# Super-Resolution Image Reconstruction using a GAN-based approach: application on Dermatology

## Prerequisites
You will need to install the following packages before you can use the code:

Albumentations
MatplotLib
NumPy
OpenCV
Pytorch
Pandas
PyRadiomics
Scikit-Image
Scikit-Learn
SimpleITK
Tqdm

It is suggested to create a new conda environment and install these packages using conda (or pip).

## dermasr:
The dermasr package provides a set of classes and methods for performing image processing, data augmentation, and degradation algorithms on skin images. This package includes the following classes:
    **artifacts**: contains the class Artifacts that can be used to handle some common image artifacts typically present in skin images.
    **augmentation**: contains the class Augmentation that can generate augmented images.
    **paired**: contains the class Paired that can generate either low-resolution images via a degradation pipeline or high-resolution images via an enhancement pipeline.


### cpp:
It contains the Artifacts functions implemented in C++.

## tutorial:
It contains a set of guides and tutorials on how to use the 'dermasr' package and replicate the most important parts of the pipeline.

## texture analysis:
It contains codes to run the 'Texture Analysis'.

## iqa:
Folder containing codes to run the 'Image Quality Assessment' on Matlab

## pipeline:
It contains codes that were used during the implementation of the Preprocessing pipeline for this thesis work.

## backup:
It contains intermediate and trial codes that were not used anymore after the completion of the project.



