import tensorflow as tf 
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#`x_test = mnist.test.images[:100,:]
labels = mnist.test.labels[:1000,:]

sess = tf.Session()

def sample_noise(batch_size, dim):
    
    random_noise = tf.random_uniform(maxval=1,minval=-1,shape=[batch_size, dim])    
    return random_noise

def generator_conv(z,c,Reuse=True):
    with tf.variable_scope("generator_conv", reuse=Reuse):
        keep_prob = 0.5
        #x = inputs
        x = z
        d1 = 4
        d2 = 256
        
        x = tf.concat(axis=1, values=[x, c])
        x = tf.layers.dense(x, units=d1*d1*d2, activation=activation)
        x = tf.layers.dropout(x, keep_prob)     
        
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[7, 7])
        
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)

        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=activation)

        x = tf.tanh(x)
        x = tf.reshape(x,shape=[-1,784])
        return x


   
z = tf.placeholder(tf.float32, shape=[None, 96])
y_dim = 10 
y = tf.placeholder(tf.float32, shape=[None, y_dim])


G_sample = generator_conv(z,y,Reuse=False)
    


def generate_fake_archive(labels,N_of_data):
    
    Load_generator()
    
    labels = labels[:N_of_data]
    noise = sample_noise(N_of_data,96) 
    Generated_samples = sess.run(G_sample,feed_dict={y:labels, z:noise})
    return Generated_samples

def Save_generator(G_vars):
    saver = tf.train.Saver(var_list=G_vars)
    saver.save(sess, 'my_generator_model.ckpt')
    print('salvo il generatore')
    

def Load_generator():
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_conv'))
    saver.restore(sess, 'my_generator_model.ckpt')

def activation(x, alpha=0.01):
    a = tf.maximum(x, alpha*x)
    return a



Generated_archive = generate_fake_archive(labels,10)

print('Generated archive shape: ', np.shape(Generated_archive))
    


    
