import tensorflow as tf

def encoder(input_data):
    with tf.name_scope("Encoder"):
        fc1 = tf.layers.dense(input_data,512, activation=tf.nn.leaky_relu,name = 'fc1_encoder')   
        fc2 = tf.layers.dense(fc1,256, activation=tf.nn.leaky_relu,name = 'fc2_encoder') 
        mu = tf.layers.dense(fc2,128, activation=None,name = 'fc3_mu_encoder')
        '''sigma is always a postive value, it make the gradint is always postive,not good'''
        '''soulion: network out put is log_sigma, sigma become exp(log_sigma)'''
        log_sigma = tf.layers.dense(fc2,128, activation=None,name = 'fc3_log_sigma_encoder')        
    return mu,log_sigma

def decoder(mu,log_sigma):
    with tf.name_scope("Latent_code_distrbution_sampling"):
        sigma = tf.exp(log_sigma)
        sample_RD = tf.random_normal(tf.shape(sigma), name = 'sample_RD') #random sample from the laten code distrbution
        latent_code = mu +  tf.exp(0.5 * log_sigma) * sample_RD
    with tf.name_scope("Decoder"):
        fc1 = tf.layers.dense(latent_code,256, activation=tf.nn.leaky_relu,name = 'fc1_decoder') 
        fc2 = tf.layers.dense(fc1,512, activation=tf.nn.leaky_relu,name = 'fc2_decoder') 
        output = tf.layers.dense(fc2,784, activation=None,name = 'output_decoder') 
    return output
