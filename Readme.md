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
Run the eval.py and eval_genresult.py will show some result from model . 

- Open the *eval.py* and *eval_genresult.py* to check the argparse setting.
- Using the argparse for evaluation parameter setting.
- Start two evualation respectively.
```Shell
python eval.py
```
```Shell
python eval_genresult.py
```

## Performance
---
- I train this model about 100 epoch.
- Current performance(RMSE):
- The part visualization result(input the random value to the decoder)

    <p align="center"><img src="" alt=" "  height='230px' width='230px'></p> 
    
- Something special

I use the L2 loss as the reconstruction loss term in VAE in the first time training, but I got the imperfect result is shown as below(50 epochs).

<p align="center"><img src="https://github.com/yulinliutw/VAE-with-Mnist/blob/master/expimg/exp_l2lossonly.png" alt=" "  height='230px' width='230px'></p>

   It seems like this loss only can make the model roughly distinguish the foreground and background part, I think that's because the curve of L2 loss is very smooth, so the  gradient for optimization will become very small when the output become close to the ground truth, then model may stop to learn early, in this case, it's not good.
   Thus I  find out some researcher use the cross entropy loss to slove this problem, this loss provide the steep curve may will handle this problem,and the value in our data is between 0~1, it's legal to use this loss, the training result is shown in  *eval.py*.

