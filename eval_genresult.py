import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from model import decoder
import os

'''init setting'''
parser = argparse.ArgumentParser(description='VAE_train: Inference Parameters')
parser.add_argument('--load_weight_dir',
                    default = '', 
                    help    = 'Path to folder of loading weight')
'''gobal setting'''
global args
args = parser.parse_args()

'''prepare model'''
mu = tf.placeholder(tf.float32, shape = [ None, 128])
log_sigma = tf.placeholder(tf.float32, shape = [ None, 128])
output = decoder(mu,log_sigma)


'''show some generation result'''  
current_result = sess.run([output], feed_dict = { mu: [] , log_sigma : [] }) 
current_result = np.reshape(current_result,(-1,28,28))
plt.figure()
plt.title('VAE generation outoput')     
for img_idx in range(100):
    plt.subplot(10,10,img_idx+1)
    plt.imshow(current_result[img_idx,:,:], cmap='gray')        
plt.show() 