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
                    default = '', 
                    help    = 'Path to folder of loading weight')
parser.add_argument('--gpuid',
                    default = 0,
                    type    = int,
                    help    = 'GPU device ids (CUDA_VISIBLE_DEVICES)')
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
RMSE = tf.sqrt(tf.reduce_mean(tf.pow(output-data,2)))

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
for ep in range(args.epoch) : 
    for step in range(int(mnist.train.images.shape[0]/args.batch)) : 
        batch_x, batch_y = mnist.train.next_batch(args.batch)
        batch_x_val, batch_y_val = mnist.validation.next_batch(args.batch)
        _= sess.run([optimizer], feed_dict = { data: batch_x })  
        
'''show some rescontrust result'''      
current_result = sess.run([output], feed_dict = { data: batch_x }) 
current_result = np.reshape(current_result,(-1,28,28))
plt.figure()
plt.title('VAE outoput')     
for img_idx in range(6):
    plt.subplot(3,2,img_idx+1)
    plt.imshow(current_result[img_idx,:,:], cmap='gray')        
plt.show() 
filename = os.path.join(args.save_weight_dir,'ep_'+str(ep+1),'checkpoint_ep'+str(ep+1)+'.ckpt')


'''show some generation result'''  
current_result = sess.run([output], feed_dict = { data: batch_x }) 
current_result = np.reshape(current_result,(-1,28,28))
plt.figure()
plt.title('VAE outoput')     
for img_idx in range(6):
    plt.subplot(3,2,img_idx+1)
    plt.imshow(current_result[img_idx,:,:], cmap='gray')        
plt.show() 
filename = os.path.join(args.save_weight_dir,'ep_'+str(ep+1),'checkpoint_ep'+str(ep+1)+'.ckpt')
