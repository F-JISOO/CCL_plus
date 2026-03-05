# [TPAMI] Continuous Contrastive Learning for Realistic Long-Tailed Semi-Supervised Recognition
Long-Tailed Semi-Supervised Recognition

This is PyTorch implementation of ***Continuous Contrastive Learning for Realistic***
***Long-Tailed Semi-Supervised Recognition*** at TPAMI in submission. 

## Abstract

Long-tailed semi-supervised learning (LTSSL) poses a significant challenge in training models with limited labeled data exhibiting a long-tailed label distribution. Current state-of-the-art LTSSL approaches heavily rely on high-quality pseudo-labels for large-scale unlabeled data. However, these methods often neglect the impact of representations learned by the neural network and struggle with real-world unlabeled data, which typically follows a different distribution than labeled data. This paper introduces a novel probabilistic framework that unifies various recent proposals in long-tail learning. Our framework derives the class-balanced contrastive loss through Gaussian kernel density estimation. We introduce a continuous contrastive learning method, CCL, extending our framework to unlabeled data using reliable and smoothed pseudo-labels. By progressively estimating the underlying label distribution and optimizing its alignment with model predictions, we tackle the diverse distribution of unlabeled data in real-world scenarios. Extensive experiments across multiple datasets with varying unlabeled data distributions demonstrate that CCL consistently outperforms prior state-of-the-art methods, achieving over 4% improvement on the ImageNet-127 dataset. We further validate the transferability of CCL by adapting it to a CLIP-based parameter-efficient fine-tuning framework called CCL+, which can seamlessly integrate many existing LTSSL methods. We show that CCL+ is not only efficient for training, but also achieves consistent gains on more challenging datasets, i.e., Semi-Aves, Semi-Fungi, and Semi-iNaturalist. 

## Method

![VLFD_model](C:\Users\Administrator\Desktop\CCL+\VLFD_model.png)


## Requirements

- Python 3.7.13
- PyTorch 1.12.0+cu116
- torchvision
- numpy
- timm

## Dataset

The directory structure for datasets looks like:
```
datasets
├── cifar-10
├── cifar-100
├── Semi-Aves
├── Semi-Fungi
├── Semi-iNaturalist
```

## PEFT Framework: A Unified Approach for Efficient LTSSL

The **PEFT (Parameter-Efficient Fine-Tuning)** framework we propose integrates multiple **Long-Tailed Semi-Supervised Learning (LTSSL)** methods, enabling highly efficient training. Our approach ensures that the benefits of existing LTSSL methods can be leveraged without requiring large-scale re-training or excessive computational resources. Through the PEFT framework, we achieve:

"We can add the `algorithm` parameter in the training command to specify the training method used."

## Usage

Train our proposed CCL+ for different settings.

For CIFAR-10:

```
# run N1 setting with VPT
python main.py --cfg configs/peft/cifar10_N1.yaml vpt_deep True

# run N2 setting with VPT
python main.py --cfg configs/peft/cifar10_N2.yaml vpt_deep True

# run N4 setting with VPT
python main.py --cfg configs/peft/cifar10_N4.yaml vpt_deep True

# run N2-uniform setting with VPT
python main.py --cfg configs/peft/cifar10_k1.yaml vpt_deep True

# run N2-reversed setting with VPT
python main.py --cfg configs/peft/cifar10_k2.yaml vpt_deep True
```
For CIFAR-100:

```
# run N4 setting with VPT
python main.py --cfg configs/peft/cifar100_N4.yaml vpt_deep True

# run N25 setting with VPT
python main.py --cfg configs/peft/cifar100_N25.yaml vpt_deep True

# run N100 setting with VPT
python main.py --cfg configs/peft/cifar100_N100.yaml vpt_deep True

# run N25-uniform setting with VPT
python main.py --cfg configs/peft/cifar100_k1.yaml vpt_deep True

# run N25-reversed setting with VPT
python main.py --cfg configs/peft/cifar100_k2.yaml vpt_deep True
```
For Semi-Aves:

```
# run Uin setting with VPT
python main.py --cfg configs/peft/semi_aves.yaml vpt_deep True

# run Uout setting with VPT
python main.py --cfg configs/peft/semi_aves_out.yaml vpt_deep True
```

For Semi-Fungi:

```
# run Uin setting with VPT
python main.py --cfg configs/peft/semi_fungi.yaml vpt_deep True

# run Uout setting with VPT
python main.py --cfg configs/peft/semi_fungi_out.yaml vpt_deep True
```

For Semi-iNaturalist:

```
# run Uin setting with VPT
python main.py --cfg configs/peft/inat.yaml vpt_deep True

# run Uout setting with VPT
python main.py --cfg configs/peft/inat_out.yaml vpt_deep True
```

## Acknowledge

We thank the authors of the [PEL](https://github.com/shijxcs/PEL) for making their code available to the public.
