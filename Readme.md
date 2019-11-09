# Variational Autoencoder(VAE) with Mnist
---

This model was trained on Mnist, finally, the model take the random latent variables as input, them its output try to generate the handwritten numeral image.

The VAE model is a interesting and amazing deep learning technology, it assum that the data can be recontrust by some latent vector, which means that the latent vector describe the basic and mainly properties of the data, all the data actually is construct by this properties. In the VAE, every properties will try to form by a normal distribution which make it easy to control and do the sample.

*you can google the Variational Autoencoder(VAE), it may help you more understand this theory.*

It is authored by **YU LIN LIU**.

### Table of Contents
- <a href='#model-architecture'>Model Architecture</a>
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training'>Training</a>
- <a href='#evaluation'>Evaluation</a>
- <a href='#performance'>Performance</a>


## Model Architecture
---
During training, the network architecture is an autoencoder, encoder and decoder are  the fully connect neural network, for the latent code setting, I the using 128 control variable(128 means and 128 sigmas).

In the testing time, it only content the decoder part.

## Installation
---
- Install the [Tensorflow](https://www.tensorflow.org/) by selecting your environment on the website.
- Clone this repository.

## Datasets
---
#### MNIST 
This dataset is a large database of handwritten digits that is commonly used for training various image processing systems. 

Current time, the tensorflow library can directly provide this dataset, we can just directly use it.  
**For the detail of this dataset in tensorflow, you can check this** [link](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.7/tensorflow/g3doc/tutorials/mnist/beginners/index.md).

## Training
---
- Open the *train.py* and check the *argparse* setting to understand the training parameters.
- Using the argparse for training parameter setting.
	* Note: we provide the pretrain weight in *./better_weight*, you can load it by setting the *load_weight_dir* parameter.
- Start the training.
```Shell
python train.py
```
--Note: 
I provide the tensorboard visualization function, it will visualize some evaluation in training and validation batch datas, you can use it to check traing history during or after training.
```Shell
tensorboard --logdir=run1:"./log/train",run2:"./log/val"
```

## Evaluation
---
Run the eval.py will show some result from model . 

- Open the *eval.py* and check the argparse setting to understand the evaluation parameters.
- Using the argparse for evaluation parameter setting.
- Start the evualation.
```Shell
python eval.py
```

## Performance
---
- I train this model about 40 epoch.
- Current performance(RMSE):
- The training history 
- The part visualization result
- Something special

