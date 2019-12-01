import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from model import encoder,decoder
import os

'''init setting'''
parser = argparse.ArgumentParser(description='VAE_train: Inference Parameters')
parser.add_argument('--batch',
                    type=int,
                    default=100,
                    help='testing batch setting')

parser.add_argument('--load_weight_dir',
                    default = './weight_train/ep_50/checkpoint_ep50.ckpt-550', 
                    help    = 'Path to folder of loading weight')
'''gobal setting'''
global args
args = parser.parse_args()

'''import MNIST data'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''prepare model'''
data = tf.placeholder(tf.float32, shape = [ None, 784 ])
mu,log_sigma = encoder(data)
output = decoder(mu,log_sigma)
    
'''define the loss''' 
RMSE = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.sigmoid(output)-data),reduction_indices = 1)/784),name = 'RMSE')

'''init the model'''
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


'''evualation procedure'''
'''caculate the RMSE'''
for step in range(int(mnist.test.images.shape[0]/args.batch)) : 
    batch_x, batch_y = mnist.test.next_batch(args.batch)        
    rmse= sess.run([RMSE], feed_dict = { data: batch_x })  
print('current rmse:'+str(rmse))    
        
'''show some rescontrust result'''      
current_result = sess.run([tf.sigmoid(output)], feed_dict = { data: batch_x }) 
current_result = np.reshape(current_result,(-1,28,28))
plt.figure()
plt.title('VAE rescontrust outoput')     
for img_idx in range(100):
    plt.subplot(10,10,img_idx+1)
    plt.imshow(current_result[img_idx,:,:], cmap='gray')      
    plt.axis('off')
plt.show() 




