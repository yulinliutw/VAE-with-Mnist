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
                    default = './weight_train/ep_50/checkpoint_ep50.ckpt-550', 
                    help    = 'Path to folder of loading weight')
'''gobal setting'''
global args
args = parser.parse_args()

'''prepare model'''
mu = tf.placeholder(tf.float32, shape = [ None, 128])
log_sigma = tf.placeholder(tf.float32, shape = [ None, 128])
output = decoder(mu,log_sigma)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
pretrain_reader = tf.train.NewCheckpointReader(args.load_weight_dir)
model_list = tf.global_variables()
model_pretrain_list = [var  for var in model_list if pretrain_reader.has_tensor(var.name.split(':')[0])] #load the pretrain layer which appear in the current model
model_loder = tf.train.Saver(model_pretrain_list)    
model_loder.restore(sess,args.load_weight_dir)
print('===================================')
print('load pre_train weight successfully')
print('===================================')

'''show some generation result'''  
current_result = sess.run([tf.sigmoid(output)], feed_dict = { mu: np.random.uniform(low=-1, high=1, size=(100, 128)) , log_sigma : np.random.uniform(low=-1, high=1, size=(100, 128)) }) 
current_result = np.reshape(current_result,(-1,28,28))
plt.figure()
plt.title('VAE generation outoput')     
for img_idx in range(100):
    plt.subplot(10,10,img_idx+1)
    plt.imshow(current_result[img_idx,:,:], cmap='gray')   
    plt.axis('off')     
plt.show() 