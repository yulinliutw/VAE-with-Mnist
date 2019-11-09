import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from model import encoder,decoder
import os

'''init setting'''
parser = argparse.ArgumentParser(description='VAE_train: Inference Parameters')
parser.add_argument('--epoch',
                    type = int,
                    default = 50,
                    help = 'training epoch setting')
parser.add_argument('--batch',
                    type = int,
                    default = 100,
                    help  ='training batch setting')
parser.add_argument('--learning_rate',
                    type = float,
                    default = 0.0002,
                    help = 'learning rate setting')

parser.add_argument('--save_weight_dir',
                    default = './weight_train_test/',
                    help    = 'Path to folder of saving weight')
parser.add_argument('--log_dir',
                    default = './log/',
                    help = 'the path to save the training record')

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
with tf.name_scope("Loss_function"):   
    loss_l2 = tf.reduce_mean(tf.norm(output-data, ord='euclidean',axis = 1),name = 'loss_l2')
    loss_bce = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = output, labels = data), reduction_indices = 1),name = 'loss_bce') #Optmize the Ez∼Q[logP(X|z)] , just check the paper 
    loss_KLD = tf.reduce_mean(0.5 * tf.reduce_sum(tf.exp(log_sigma) + tf.pow(mu, 2) - log_sigma - 1, reduction_indices = 1),name = 'loss_KLD') #Optmize the D[Q(z|X)∥P(z)] , just check the paper 
    loss_total = 0.5 * loss_l2 + 0.5 * loss_bce + loss_KLD   
    RMSE = tf.sqrt(tf.reduce_mean(tf.pow(output-data,2)),name = 'RMSE')
    '''add the loss to the tensorboard'''
    tf.summary.scalar('loss_l2',loss_l2)
    tf.summary.scalar('loss_bce', loss_bce)
    tf.summary.scalar('loss_KLD', loss_KLD)    
    tf.summary.scalar('loss_total', loss_total)
    tf.summary.scalar('RMSE_loss', RMSE)
    
'''define the optimizer'''
optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss_total)

'''init the model'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())
try:
    pretrain_reader = tf.train.NewCheckpointReader(args.load_weight_dir)
    model_list = tf.global_variables()
    model_pretrain_list = [var  for var in model_list if pretrain_reader.has_tensor(var.name.split(':')[0])] #load the pretrain layer which appear in the current model
    model_loder = tf.train.Saver(model_pretrain_list)    
    model_loder.restore(sess,args.load_weight_dir)
    print('===================================')
    print('load pre_train weight successfully')
    print('===================================')
except:    
    print('===================================')
    print('       random init the weight      ')
    print('===================================')

'''training procedure'''

merged = tf.summary.merge_all()
'''create two summary writer for showing the train and test together'''
'''command : tensorboard --logdir=run1:"./log/train",run2:"./log/val" '''
summary_writer_train = tf.summary.FileWriter(args.log_dir+'/train', sess.graph)
summary_writer_val = tf.summary.FileWriter(args.log_dir+'val')

saver = tf.train.Saver(max_to_keep = 0)
for ep in range(args.epoch) : 
    for step in range(int(mnist.train.images.shape[0]/args.batch)) : 
        batch_x, batch_y = mnist.train.next_batch(args.batch)
        batch_x_val, batch_y_val = mnist.validation.next_batch(args.batch)
        _= sess.run([optimizer], feed_dict = { data: batch_x })  
        
        '''show some current prediction & save weight'''
        if(((step+1)%10)==0):   
            '''visualize the batch loss to roughly analysis the trend of the loss '''
            summary_train,l2_loss,bce_loss,kld_loss,total_loss,rmse= sess.run([merged,loss_l2,loss_bce,loss_KLD,loss_total,RMSE], feed_dict = { data: batch_x })  
            summary_writer_train.add_summary(summary_train,int(mnist.train.images.shape[0]/args.batch)*ep+step+1)
            summary_val,l2_loss,bce_loss_val,kld_loss_val,total_loss_val,rmse_val= sess.run([merged,loss_l2,loss_bce,loss_KLD,loss_total,RMSE], feed_dict = { data: batch_x_val })
            summary_writer_val.add_summary(summary_val,int(mnist.train.images.shape[0]/args.batch)*ep+step+1)
            current_result = sess.run([output], feed_dict = { data: batch_x }) 
            current_result = np.reshape(current_result,(-1,28,28))
            plt.figure()
            plt.title('VAE outoput')     
            for img_idx in range(6):
                plt.subplot(3,2,img_idx+1)
                plt.imshow(current_result[img_idx,:,:], cmap='gray')        
            plt.show() 
            filename = os.path.join(args.save_weight_dir,'ep_'+str(ep+1),'checkpoint_ep'+str(ep+1)+'.ckpt')
            save_path = saver.save(sess,filename, global_step=step+1)

summary_writer_train.close()            
summary_writer_val.close()