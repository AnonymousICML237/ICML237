ICML2019 Submission 237
==

This repository contains the source code for ICML2019 submission 237 'Diversified Progressive Layerwise Adversarial Training for Improving'. We give the codes for VGG-16 on CIFAR-10.


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

Model Robustness Evaluation
--
* Empirical Worst Case Decision Boundary Distance<br>
sh db.sh
* Empirical Noise Insensitivity<br>
sh eni.sh
* Corruption Robustness Evaluation<br>
sh cpre1.sh and sh cpre2.sh

Comparative Adversarial Defense Methods
--
* Original adversarial training ([Goodfellow et al.](https://arxiv.org/pdf/1412.6572.pdf))<br>
sh OAT.sh
* New adversarial training ([(Kurakin et al.](https://arxiv.org/pdf/1607.02533.pdf))<br>
sh NAT.sh
* Randomization ([Xie et al.](https://arxiv.org/pdf/1711.01991))<br>
sh RAND.sh
* Ensemble adversarial training ([Tramer et al.](https://arxiv.org/pdf/1705.07204.pdf))<br>
sh EAT.sh

Run Adversarial Attack Methods
--
sh attack.sh
