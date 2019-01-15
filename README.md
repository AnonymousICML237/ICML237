ICML 237
==

This repository contains the source code for ICML2019 submission No.237 'Diversified Progressive Layerwise Adversarial Training for Improving'. We give the codes for VGG-16 on CIFAR-10.


Dependencies
--
This library uses Pytorch to accelerate graph computations performed by many machine learning models. Therefore, installing TensorFlow is a pre-requisite.<br>
Installing Pytorch will take care of all other dependencies like numpy and scipy.

Train models
--
sh train.sh

Test models
--
sh test.sh

Robustness evaluation
--
### Empirical Worst Case Decision Boundary Distance
sh db.sh
### Empirical Adversarial Insensitivity
sh eai.sh
### Corruption and Perturbation Robustness Evaluation
sh cpre.sh
