import tensorflow as tf 
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#`x_test = mnist.test.images[:100,:]
labels = mnist.test.labels[:1000,:]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def funzione_test():
    return (np.random.rand(128,10),np.random.rand(128,32))
a = []
b = []
a,b = zip(*[funzione_test() for _ in range(5)])

c_1 = np.reshape(a,[128*5,10])
c_2 = np.reshape(b,[128*5,32])

import TF_layers 
from TF_layers import Batchnorm,conv_transpose

def define_placeholders(Noise_dim=96,label_shape=10):
     z = tf.placeholder(tf.float32, shape=[None, Noise_dim])
     y = tf.placeholder(tf.float32, shape=[None, label_shape])
     return z,y
 
def Run_images_generation(labels,z=None,y=None,N_to_generate=128,First_time=True,Noise_dim=80,label_shape=10):
    if z==None:
        z = tf.placeholder(tf.float32, shape=[None, Noise_dim])
    if y==None:
        y = tf.placeholder(tf.float32, shape=[None, label_shape])
    if First_time == True:
        G_sample = generator_conv(z,y,False) 
    if First_time != True:
        G_sample = generator_conv(z,y,True)    

    Generated_archive = generate_fake_archive(G_sample,y,z,labels,N_to_generate,First_time)
    
    #print('Generated archive shape: ', np.shape(Generated_archive))
    #fig = disp_images(Generated_archive[:50])
    #plt.show()
    return Generated_archive

def sample_noise(batch_size, dim):
    
    random_noise = tf.random_uniform(maxval=1,minval=-1,shape=[batch_size, dim])    
    return random_noise



def generator_conv(x,c,Reuse=True):
   
    
    batch_size = np.shape(x)[0]
    
    o_h = 64
    o_w = 64
        
    s_h = np.array([o_h/16, o_h/8, o_h/4, o_h/2, o_h])
    s_w = np.array([o_w/16, o_w/8, o_w/4, o_w/2, o_w])
    
    
    s_h = s_h.astype(int)
    s_w = s_w.astype(int)
  
    with tf.variable_scope("generator_conv", reuse=Reuse):
        
       #x = inputs
        
        d1 = 4
        d2 = 128
        
    
        init = tf.contrib.layers.xavier_initializer()
        
        x = tf.concat(axis=1, values=[x, c])
        
        x = tf.layers.dense(x, units=d1*d1*d2)
        x = Batchnorm('Generator.BN0', [0], x)
        x = tf.nn.leaky_relu(x)
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        
        #x = tf.image.resize_images(x, size=[s_h[0], s_w[0]])
       
        
        # layer and batch normal
        x = conv_transpose(x, [batch_size, s_h[1], s_w[1],128 ], "G_conv0",initializer=init,strides=[1,2,2,1])
        x = Batchnorm('Generator.BN1', [0,1,2], x)
        x = tf.nn.leaky_relu(x)
        
        
        x = conv_transpose(x, [batch_size, s_h[2], s_w[2], 64], "G_conv1",initializer=init,strides=[1,2,2,1])
        x = Batchnorm('Generator.BN2', [0,1,2], x)
        x = tf.nn.leaky_relu(x)
        
        
        x = conv_transpose(x, [batch_size, s_h[3], s_w[3], 64],"G_conv2",initializer=init,strides=[1,2,2,1])
        x = Batchnorm('Generator.BN3', [0,1,2], x)
        x = tf.nn.leaky_relu(x)
       
        
        x = conv_transpose(x, [batch_size, s_h[4], s_w[4], 32],"G_conv3",initializer=init,strides=[1,2,2,1])
        x = Batchnorm('Generator.BN4', [0,1,2], x)
        x = tf.nn.leaky_relu(x)
       
        
        x = conv_transpose(x, [batch_size, s_h[4], s_w[4], 3],"G_conv4",initializer=init,strides=[1,1,1,1])
    
        x = tf.tanh(x)
        
        x = tf.reshape(x,shape=[-1,4096*3])

        return x

def generate_fake_archive(G_sample,y,z,labels,N_of_data,First_time=True):
    
    if First_time == True:
        print('è la prima volta')
        #Load_generator()
    
    labels = labels[:N_of_data]
    noise = sess.run(sample_noise(N_of_data,96))
    Generated_samples = sess.run(G_sample,feed_dict={y:labels, z:noise})
    return Generated_samples

def Save_generator(G_vars):
    saver = tf.train.Saver(var_list=G_vars)
    saver.save(sess, './my_generator_model.ckpt')
    print('salvo il generatore')
    

def Load_generator():
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator_conv'))
    #saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_conv'))
    saver.restore(sess, 'my_generator_model.ckpt')

def activation(x, alpha=0.01):
    a = tf.maximum(x, alpha*x)
    return a




def disp_images(images):
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    figure = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return


_z,_y = define_placeholders()
Generated_images_regularization = Run_images_generation(labels,z=_z,y=_y)
#labels,Noise_dim=96,label_shape=10,z=None,y=None,First_time=True
#for i in range(5):
#    if i == 0:
#        Run_script(labels,z=_z,y=_y)
#    if i !=0:
#        Run_script(labels,z=_z,y=_y,First_time=True)
 
    
