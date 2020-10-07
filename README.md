
# SMILE: Semantically-guided Multi-attribute Image and Layout Editing
## SMILE - Official PyTorch Implementation [[Paper]]() [[Video]]()

<p align="left"><img width="99%" src="Figures/smile_teaser.png" /></p>

This repository provides the official PyTorch implementation of the following paper:
> **SMILE: Semantically-guided Multi-attribute Image and Layout Editing**<br>
> [Andrés Romero](https://github.com/affromero), [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en), [Radu Timofte](https://people.ee.ethz.ch/~timofter/)<br>
> CVL, ETH Zürich<br>
> https://arxiv.org/abs/
>
> **Abstract:** *Attribute image manipulation has been a very active topic since the introduction of Generative Adversarial Networks (GANs). Exploring the disentangled attribute space within a transformation is a very challenging task due to the multiple and mutually-inclusive nature of the facial images, where different labels (eyeglasses, hats, hair, identity, etc.) can co-exist at the same time. Several works address this issue either by exploiting the modality of each domain/attribute using a conditional random vector noise, or extracting the modality from an exemplary image. However, existing methods cannot handle both random and reference transformations for multiple attributes, which limits the generality of the solutions. In this paper, we successfully exploit a multimodal representation that handles all attributes, be it guided by random noise or exemplar images, while only using the underlying domain information of the target domain. We present extensive qualitative and quantitative results for facial datasets and several different attributes that show the superiority of our method. Additionally, our method is capable of adding, removing or changing either fine-grained or coarse attributes by using an image as a reference or by exploring the style distribution space, and it can be easily extended to head-swapping and face-reenactment applications without being trained on videos.*

## Results
SMILE can manipulate a source image into an output image reflecting the attribute and style (e.g., eyeglasses, hat, hair, etc.) of a different person.

<p align="left"><img width="99%" src="Figures/video_teaser.gif" /></p>

Eyeglasses                       |  Hat
:-------------------------------:|:-------------------------------:
![](Figures/out_eyeglasses.gif)  |  ![](Figures/out_hat.gif)

Hair                             |  Bangs
:-------------------------------:|:-------------------------------:
![](Figures/out_hair.gif)        |  ![](Figures/out_bangs.gif)

Puppet                           |  Head Swap
:-------------------------------:|:-------------------------------:
![](Figures/out_puppet.gif)        |  ![](Figures/out_video.gif)

## Overview of the method
<p align="left"><img width="99%" src="Figures/overview.png" /></p>

## Installation
Code, pretrained models and usage examples will be updated soon. Please stay tuned.

## Citation
If you find this work is useful for your research, please cite our paper:
```
@article{romero2020smile,
  title={SMILE: Semantically-guided Multi-attribute Image and Layout Editing},
  author={Andr\'es Romero and Luc Van Gool and Radu Timofte},
  journal={arXiv preprint arXiv:--},
  year={2020}
}
```
