from __future__ import print_function, division
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import sys

import tensorflow as tf
import os
from os.path import join, getsize
import tensorflow.contrib.layers as tcl



BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(BASE_DIR, 'Desktop\working_gan_thesis\EuroSat_GAN'))

import EuroSatArchive
from EuroSatArchive import load_batch



critich = 5 ; 
batch_size = 128
noise_dim = 100
DIM = 64

o_h = 64
o_w = 64
        
s_h = np.array([o_h/16, o_h/8, o_h/4, o_h/2, o_h])
s_w = np.array([o_w/16, o_w/8, o_w/4, o_w/2, o_w])
    
s_h = s_h.astype(int)
s_w = s_w.astype(int)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

FLAGS = tf.app.flags.FLAGS

tf.reset_default_graph()

init = tf.contrib.layers.xavier_initializer()


def gaussian_noise_layer(input_layer, std=0.2):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise


def linear(input, output_dim, scope=None, stddev=1.0):

    with tf.variable_scope(scope or 'linear'):

        w = tf.get_variable(

            'w',

            [input.get_shape()[1], output_dim],

            initializer=tf.random_normal_initializer(stddev=stddev)

        )

        b = tf.get_variable(

            'b',

            [output_dim],

            initializer=tf.constant_initializer(0.0)

        )

        return tf.matmul(input, w) + b


def minibatch_discrimination(input, num_kernels=100, kernel_dim=3):

    #x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    x = tf.layers.dense(input, units=num_kernels*kernel_dim) #        x = tf.layers.dense(x, units=d1*d1*d2)

    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))

    diffs = tf.expand_dims(activation, 3) -  tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)

    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)

    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)

    return tf.concat([input, minibatch_features], 1)



#saver = tf.train.Saver()
_params = {}
_param_aliases = {}

def param(name, *args, **kwargs):
    """
    A wrapper for `tf.Variable` which enables parameter sharing in models.
    
    Creates and returns theano shared variables similarly to `tf.Variable`, 
    except if you try to create a param with the same name as a 
    previously-created one, `param(...)` will just return the old one instead of 
    making a new one.

    This constructor also adds a `param` attribute to the shared variables it 
    creates, so that you can easily search a graph for all params.
    """

    if name not in _params:
        kwargs['name'] = name
        param = tf.Variable(*args, **kwargs)
        param.param = True
        _params[name] = param
    result = _params[name]
    i = 0
    while result in _param_aliases:
        # print 'following alias {}: {} to {}'.format(i, result, _param_aliases[result])
        i += 1
        result = _param_aliases[result]
    return result

def Batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True):
    if ((axes == [0,1,2]) or (axes == [0,2])) and fused==True:
        if axes==[0,2]:
            inputs = tf.expand_dims(inputs, 3)
        # Old (working but pretty slow) implementation:
        ##########

        # inputs = tf.transpose(inputs, [0,2,3,1])

        # mean, var = tf.nn.moments(inputs, [0,1,2], keep_dims=False)
        # offset = lib.param(name+'.offset', np.zeros(mean.get_shape()[-1], dtype='float32'))
        # scale = lib.param(name+'.scale', np.ones(var.get_shape()[-1], dtype='float32'))
        # result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-4)

        # return tf.transpose(result, [0,3,1,2])

        # New (super fast but untested) implementation:
        offset = param(name+'.offset', np.zeros(inputs.get_shape()[3], dtype='float32'))
        scale = param(name+'.scale', np.ones(inputs.get_shape()[3], dtype='float32'))
        
        moving_mean = param(name+'.moving_mean', np.zeros(inputs.get_shape()[3], dtype='float32'), trainable=False)
        moving_variance = param(name+'.moving_variance', np.ones(inputs.get_shape()[3], dtype='float32'), trainable=False)

        def _fused_batch_norm_training():
            return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-5, data_format='NHWC')
        def _fused_batch_norm_inference():
            # Version which blends in the current item's statistics
            batch_size = tf.cast(tf.shape(inputs)[0], 'float32')
            mean, var = tf.nn.moments(inputs, [1,2], keep_dims=True)
            mean = ((1./batch_size)*mean) + (((batch_size-1.)/batch_size)*moving_mean)[None,:,None,None]
            var = ((1./batch_size)*var) + (((batch_size-1.)/batch_size)*moving_variance)[None,:,None,None]
            return tf.nn.batch_normalization(inputs, mean, var, offset[None,:,None,None], scale[None,:,None,None], 1e-5), mean, var

            # Standard version
            # return tf.nn.fused_batch_norm(
            #     inputs,
            #     scale,
            #     offset,
            #     epsilon=1e-2, 
            #     mean=moving_mean,
            #     variance=moving_variance,
            #     is_training=False,
            #     data_format='NCHW'
            # )

        if is_training is None:
            outputs, batch_mean, batch_var = _fused_batch_norm_training()
        else:
            outputs, batch_mean, batch_var = tf.cond(is_training,
                                                       _fused_batch_norm_training,
                                                       _fused_batch_norm_inference)
            if update_moving_stats:
                no_updates = lambda: outputs
                def _force_updates():
                    """Internal function forces updates moving_vars if is_training."""
                    float_stats_iter = tf.cast(stats_iter, tf.float32)

                    update_moving_mean = tf.assign(moving_mean, ((float_stats_iter/(float_stats_iter+1))*moving_mean) + ((1/(float_stats_iter+1))*batch_mean))
                    update_moving_variance = tf.assign(moving_variance, ((float_stats_iter/(float_stats_iter+1))*moving_variance) + ((1/(float_stats_iter+1))*batch_var))

                    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                        return tf.identity(outputs)
                outputs = tf.cond(is_training, _force_updates, no_updates)

        if axes == [0,2]:
            return outputs[:,:,:,0] # collapse last dim
        else:
            return outputs
    else:
        # raise Exception('old BN')
        # TODO we can probably use nn.fused_batch_norm here too for speedup
        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        shape = mean.get_shape().as_list()
        if 0 not in axes:
            print("WARNING ({}): didn't find 0 in axes, but not using separate BN params for each item in batch".format(name))
            shape[0] = 1
        offset = param(name+'.offset', np.zeros(shape, dtype='float32'))
        scale = param(name+'.scale', np.ones(shape, dtype='float32'))
        result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)


        return result


def init_variables():
    sess.run(tf.global_variables_initializer())

def disp_images(images):
    N_channels = 3 ; 
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1]/N_channels)))
    figure = plt.figure(figsize=(sqrtn+2, sqrtn+2))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg, N_channels]))
    return

def preprocess(x):
    return 2*x-1.0

def deprocess(x):
    return (x+1.0) / 2.0

def rel_error(x):
    return np.max(np.abs(x-y)/ (np.maximum(1e-8. np.abs(x) + np.abs(y))))

def count_params():
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count

def session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    return session

#answers = np.load('gan-checks-tf.npz')

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
minibatch,minbatch_y = load_batch(batch_size = batch_size)
disp_images(minibatch[0:16])
print("finito di stamprare immagini del dataset")

def activation(x, alpha=0.01):
    a = tf.maximum(x, alpha*x)
    return a


def sample_noise(batch_size, dim):
    random_noise = tf.random_uniform(maxval=1,minval=-1,shape=[batch_size, dim])
    return random_noise

def get_solvers(lr= 3e-4, beta1=0.5,beta2=0.9):
    d_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
    g_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
    return d_solver,g_solver 

    
def _variable_on_cpu(name, shape, initializer):

  """Helper to create a Variable stored on CPU memory.



  Args:

    name: name of the variable

    shape: list of ints

    initializer: initializer for Variable



  Returns:

    Variable Tensor

  """
  with tf.variable_scope("_variable_on_cpu", reuse=tf.AUTO_REUSE):

      dtype =tf.float32 #tf.float16 if FLAGS.use_fp16 else tf.float32
      var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  
  return var
    
def _variable_with_weight_decay(name, shape, stddev, wd):

  """Helper to create an initialized Variable with weight decay.



  Note that the Variable is initialized with a truncated normal distribution.

  A weight decay is added only if one is specified.



  Args:

    name: name of the variable

    shape: list of ints

    stddev: standard deviation of a truncated Gaussian

    wd: add L2Loss weight decay multiplied by this float. If None, weight

        decay is not added for this Variable.



  Returns:

    Variable Tensor

  """

  dtype =tf.float32 #tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

  if wd is not None:

    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')

    tf.add_to_collection('losses', weight_decay)

  return var

def huber_loss(labels, predictions, delta=1.0):

    residual = tf.abs(predictions - labels)

    condition = tf.less(residual, delta)

    small_res = 0.5 * tf.square(residual)

    large_res = delta * residual - 0.5 * tf.square(delta)

    return tf.where(condition, small_res, large_res)

def conv2d(x, inputFeatures, outputFeatures, name,initializer,strides = [1,1,1,1]):

    with tf.variable_scope(name):

        w = tf.get_variable("w",[5,5,inputFeatures, outputFeatures], initializer=initializer)

        b = tf.get_variable("b",[outputFeatures], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(x, w, strides=strides, padding="SAME") + b

        return conv

def conv_transpose(x, outputShape,name,initializer,strides=[1,2,2,1]):

    with tf.variable_scope(name):

        # h, w, out, in

        w = tf.get_variable("w",[5,5, outputShape[-1], x.get_shape()[-1]], initializer=initializer)

        b = tf.get_variable("b",[outputShape[-1]], initializer=tf.constant_initializer(0.0))

        convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=strides) +b 

        return convt
    
def generator_conv(x,c,Reuse=True):
   
    #inputs = tf.concat(axis=1, values=[z, c])
    #print('shape z: ',np.shape(z))
    #print('shape c: ',np.shape(c))
    #print('shape inputs: ',np.shape(inputs))
 
    #momentum = 0.99
    
  
    with tf.variable_scope("generator_conv", reuse=Reuse):
                
        x = tf.concat(axis=1, values=[x, c])
          
        x = tf.layers.dense(x, units=4*4*64) #        x = tf.layers.dense(x, units=d1*d1*d2)
        x = tf.nn.leaky_relu(x)
        
        x = tf.reshape(x, shape=[-1, 4, 4, 64])
               
        
        # layer and batch normal
        x = conv_transpose(x, [batch_size, s_h[2], s_w[2], 64], "G_conv0",initializer=init,strides=[1,4,4,1])
        x = tf.nn.leaky_relu(x)
        
        #x = conv_transpose(x, [batch_size, s_h[2], s_w[2], 32], "G_conv1",initializer=init,strides=[1,4,4,1])
        #x = Batchnorm('Generator.BN2', [0,1,2], x)
        #x = tf.nn.leaky_relu(x)
        
        #x = conv_transpose(x, [batch_size, s_h[3], s_w[3], 16],"G_conv2",initializer=init)
        #x = Batchnorm('Generator.BN3', [0,1,2], x)
        #x = tf.nn.leaky_relu(x)
        
        #x = conv_transpose(x, [batch_size, s_h[3], s_w[3], DIM],"G_conv3",initializer=init)
        #x = Batchnorm('Generator.BN4', [0,1,2], x)
        #x = tf.nn.leaky_relu(x)
        
        x = conv_transpose(x, [batch_size, s_h[4], s_w[4], 3],"G_conv4",initializer=init,strides=[1,4,4,1])
        x = tf.tanh(x)
        
        
        x = tf.reshape(x,shape=[-1,4096*3])

        return x
    
def discriminator(x,Reuse=True):
    
    with tf.variable_scope('discriminator',reuse=Reuse):
        
        
        keep_prob = 0.5 
        init = tf.contrib.layers.xavier_initializer() 
        
        x = gaussian_noise_layer(x)
        x = tf.reshape(x, [-1,64,64,3])
        
        x = (conv2d(x, 3, 16, "D_conv0",initializer=init,strides = [1,4,4,1]))
        x = Batchnorm('Generator.BN0', [0,1,2], x)
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dropout(x, keep_prob)

        x = conv2d(x, 16, 32, "D_conv1",initializer=init,strides = [1,4,4,1])
        x = Batchnorm('Generator.BN1', [0,1,2], x)
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dropout(x, keep_prob)

        #x = conv2d(x,32, 64, "D_conv2",initializer=init,strides = [1,2,2,1])
        #x = tf.nn.leaky_relu(x)
        #x = tf.layers.dropout(x, keep_prob)

        
        #x = conv2d(x, 64, 64, "D_conv3",initializer=init,strides = [1,2,2,1])
        #if MODE != 'wgan-CT':
        #    x = Batchnorm('Discriminator.BN3', [0,1,2], x)
        #x = tf.nn.leaky_relu(x)
        #x = tf.layers.dropout(x, keep_prob)
        
        x = minibatch_discrimination(x)
       
        
        #x = conv2d(x, 64, 128, "D_conv4",initializer=init,strides = [1,1,1,1])
        #if MODE != 'wgan-CT':
        #    x = Batchnorm('Discriminator.BN4', [0,1,2], x)
        #x = tf.nn.leaky_relu(x)
        #x = tf.layers.dropout(x, keep_prob)
        
        #x = conv2d(x, 128, 128, "D_conv5",initializer=init,strides = [1,1,1,1])
        #if MODE != 'wgan-CT':
        #    x = Batchnorm('Discriminator.BN5', [0,1,2], x)
        #x = tf.nn.leaky_relu(x)
        #x = tf.layers.dropout(x, keep_prob)
        
        #x = conv2d(x, 128, 128, "D_conv6",initializer=init,strides = [1,1,1,1])
        #if MODE != 'wgan-CT':
        #    x = Batchnorm('Discriminator.BN6', [0,1,2], x)
        #x = tf.nn.leaky_relu(x)
        #x = tf.layers.dropout(x, keep_prob)
        
        x = tf.reshape(x,[-1,4*4*32])
        #x = tf.layers.dense(x, 1024,activation=activation, kernel_initializer=init,name='dense_0')
        #x = tf.layers.dropout(x, keep_prob)
        Classification = tf.layers.dense(x,10,kernel_initializer=init, name='classification')
        logits = tf.layers.dense(x,1,kernel_initializer=init, name='logits')
#                out = tf.layers.dense(x, units=10,kernel_initializer=init,activation=None)

        return logits,x,Classification

def Save_generator(G_vars):
    saver = tf.train.Saver(var_list=G_vars)
    saver.save(sess, 'my_generator_model.ckpt')
    print('salvo il generatore')
    
def Load_generator():
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_conv'))
    saver.restore(session, 'my_generator_model.ckpt')

def train(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step,
        D_extra_step,D_grad,G_grad,show_every=250, print_every=50, batch_size=128, num_epoch=10):

    
    N_EuroSatTrain_ex = 22950
    max_iter = int(N_EuroSatTrain_ex*num_epoch/batch_size)
    print("max iter: ",max_iter)
     
    #tf.summary.scalar("discriminator gradient",D_grad)
    #tf.summary.scalar("generator gradient",G_grad)
    tf.summary.scalar("Discriminator loss", D_loss)
    tf.summary.scalar("Generator loss", G_loss)
    merged_summary_op = tf.summary.merge_all()
    logs_path = '/tmp/tensorflow_logs/example/'
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    
    max_iter =  1 
    
    '''
    for it in range(50):
        minibatch,minbatch_y = load_batch(batch_size = 64)
        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch,y:minbatch_y})
        if(it % print_every == 0):
            print('Iter: {}, D: {:.4}'.format(it,D_loss_curr))
    print("save model")
    '''
    '''
    save_path = saver.save(sess, "/tmp/EuroSat_TDiscriminator_model.ckpt")
    print("Model saved in path: %s" % save_path)
    '''
    
#    init_variables()
#    saver.restore(sess, "/tmp/EuroSat_TDiscriminator_model.ckpt")
    
    
    
    for it in range(max_iter):
        
        
        for _ in range(critich):
            minibatch,minibatch_y = load_batch(batch_size=batch_size)
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch,y:minibatch_y})
            
        minibatch,minibatch_y = load_batch(batch_size=batch_size)
        _, G_loss_curr,summary = sess.run([G_train_step, G_loss,merged_summary_op], feed_dict={x: minibatch,y:minibatch_y})
        
        
        
        summary_writer.add_summary(summary,it)
        
        if(it % print_every == 0):
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
            
            
        if(it % show_every == 0):

            samples = sess.run(G_sample,feed_dict={y:minibatch_y})
            samples = deprocess(samples)
        
            
            fig = disp_images(samples[:16])
            plt.show()
            print()
            

    print('Final images')
    
    samples = sess.run(G_sample,feed_dict={y:minibatch_y})
    samples = deprocess(samples)
    fig = disp_images(samples[:16])
    plt.show()




    

x = tf.placeholder(tf.float32, [None,4096*3])
y_dim = 10 ; 
y = tf.placeholder(tf.float32, shape=[None, y_dim])

z = sample_noise(batch_size, noise_dim)

#G_sample = generator(z)
G_sample = generator_conv(z,y,Reuse=False)

G_sample_noisy = gaussian_noise_layer(G_sample)

with tf.variable_scope("") as scope:
    
    logits_real,feature_real,C_real = discriminator(preprocess(x),Reuse=False) 
    
    scope.reuse_variables() # da controllare 
    logits_fake,feature_fake,C_fake = discriminator(G_sample_noisy)
    
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator_conv')

d_solver, g_solver = get_solvers()


def wgan_loss(logits_real, logits_fake, batch_size,x,G_sample):
    d_loss =- tf.reduce_mean(logits_real) + tf.reduce_mean(logits_fake)
    g_loss =- tf.reduce_mean(logits_fake)
    
    lam = 10
    
    eps = tf.random_uniform([batch_size,1], minval=0.0,maxval=1.0)
    x_h = eps*x+(1-eps)*G_sample
    
    with tf.variable_scope("", reuse=True) as scope:
        A,_,_ = discriminator(x_h)
        grad_d_x_h = tf.gradients(A, x_h)
    
    grad_norm = tf.norm(grad_d_x_h[0], axis=-1, ord='euclidean')   #-1 per canali 
    grad_pen = tf.reduce_mean(tf.square(grad_norm-1))
    
    d_loss+=lam*grad_pen
    
    #contestual loss  
    
    return d_loss, g_loss



d_loss, g_loss = wgan_loss(logits_real, logits_fake,batch_size,x,G_sample_noisy)

s_loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=C_real,labels=y) )
d_loss += s_loss

#image matching 
#recon_weight = tf.cast(1.0, tf.float32)
#g_loss += tf.reduce_mean(huber_loss(tf.reshape(preprocess(x), [-1,64*64*3]),tf.reshape(G_sample, [-1,64*64*3])))*recon_weight




G_grad = g_solver.compute_gradients(loss=g_loss)
D_grad = d_solver.compute_gradients(loss=d_loss)

d_train_step = d_solver.minimize(d_loss, var_list=D_vars)
g_train_step = g_solver.minimize(g_loss, var_list=G_vars)
d_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
g_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator_conv')



with session() as sess:
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    train(sess,g_train_step,g_loss,d_train_step,d_loss,g_extra_step,d_extra_step,D_grad,G_grad,batch_size=batch_size,
          num_epoch=20)
    Save_generator()
# da testare

  