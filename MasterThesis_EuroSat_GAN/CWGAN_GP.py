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
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath("__file__"))


sys.path.append(os.path.join(BASE_DIR, 'Desktop\working_gan_thesis\EuroSat_GAN'))

import EuroSatArchive
from EuroSatArchive import load_batch

import Inception_score 
from Inception_score import get_inception_score
from Inception_score import _init_inception


MODE = 'wgan_Cp' #altro per gan gp
batch_size = 128
noise_dim = 128 #96
critich = 5 
Factor_M = 0 
DIM = 128
y_dim = 10 





'''
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
'''


FLAGS = tf.app.flags.FLAGS

tf.reset_default_graph()

T_t = tf.placeholder(tf.float32, shape=())
D_t= tf.placeholder(tf.float32, shape=())

def eval_confusion_matrix(labels, predictions):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=tf.argmax(labels,1), predictions=tf.argmax(predictions,1), num_classes=10)
        #cm = tf.confusion_matrix(labels=tf.argmax(batched_val_labels, 1), predictions=tf.argmax(_p, 1))
        con_matrix_sum = tf.Variable(tf.zeros(shape=(10,10), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])


        update_op = tf.assign_add(con_matrix_sum, con_matrix)

        return tf.convert_to_tensor(con_matrix_sum), update_op

def gaussian_noise_layer(input_layer, std=0.2):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def Get_dataset(Train=True,Archive_Dim=0):
    
    if Archive_Dim == 0:
        if Train==True:
            Archive_dim = 22950
        if Train==False:
            Archive_dim = 27000-22950
    if Archive_Dim != 0:        
        Archive_dim =Archive_Dim
    
    Images,Labels = load_batch(Archive_dim,Train)
    return Images,Labels

def Get_batch(Images,Labels,it,batch_size=128,Train=True):
    '''
    if Train==True:
        Data_dim = 22950
    if Train==False : 
        Data_dim = 27000-22950
    '''    
    Data_dim = Images.shape[0]    
    it = it % (int(Data_dim/128)+1)
    
    if((it+1)*batch_size>Data_dim):
        return  Images[(Data_dim-batch_size):Data_dim],Labels[(Data_dim-batch_size):Data_dim]
    
    return Images[it*batch_size:(it+1)*batch_size],Labels[it*batch_size:(it+1)*batch_size]
    

#Eurosat_data,Eurosat_labels = Get_dataset()
Eursat_data_test,Eurosat_labels_test = Get_dataset(Train=False)
a = Eursat_data_test[:128*5]
b = Eurosat_labels_test[:128*5]


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

def minibatch_discriminator(input, num_kernels=100, kernel_dim=5):
    x = linear(input, num_kernels * kernel_dim)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    print('size of matrix: ',activation)
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    
    
    a = tf.concat([input, minibatch_features],axis = 1)
    print('tensore:-------> ',a)
    return tf.concat([input, minibatch_features],axis = 1)
#saver = tf.train.Saver()merced_GAN\saved_images\\'+name)
def save_images(images,it,last=False):
    images = images*255
    images = tf.cast(images, tf.uint8)
    images = tf.reshape(images, [-1,64,64,3])
    print('feature shape: ',np.shape(images))
    
    #N_of_images = 16
        
    for i in range(16): 
        images_encode =  tf.image.encode_jpeg(images[i])
        name = 'iter'+str(it)+'NG'+str(i) +'.jpeg'
        fwrite = tf.write_file(name, images_encode)                         
        sess.run(fwrite)     
        os.rename(r'C:\Users\samuele\\' +name,r'C:\Users\samuele\Desktop\working_gan_thesis\EuroSat_GAN\saved_images\data_iterations\\'+name)

    if last==True:
        #num_of_samples = np.shape(images)[0]
        num_of_samples = 32
        for i in range(num_of_samples): 
            images_encode =  tf.image.encode_jpeg(images[i])
            name = 'iter'+str(it)+'NG'+str(i) +'.jpeg'
            fwrite = tf.write_file(name, images_encode)                         
            sess.run(fwrite)     
            os.rename(r'C:\Users\samuele\\' +name,r'C:\Users\samuele\Desktop\working_gan_thesis\EuroSat_GAN\saved_images\data\\'+name)

_params = {}
_param_aliases = {}

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
            # return tf.nn.fused_batch_norm
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
minibatch,minibatch_y = load_batch(batch_size=batch_size)

#minibatch,minbatch_y = Get_batch(Eurosat_data,Eurosat_labels,0,batch_size=128)
#minibatch,minbatch_y = load_batch(batch_size = batch_size)
disp_images(minibatch[0:16])
print("finito di stamprare immagini del dataset")



def activation(x, alpha=0.01):
    a = tf.maximum(x, alpha*x)
    return a

def sample_noise(batch_size, dim):
    random_noise = tf.random_uniform(maxval=1,minval=-1,shape=[batch_size, dim])
    
    return random_noise

def get_solvers(lr= 3e-4, beta1=0.5,beta2=0.9):
    d_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1,beta2 = beta2)
    g_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
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
    

def Find_diff_batch(Batch):
    y_shape,x_shape = np.shape(Batch) 
    diff_Batch = np.zeros(shape=[y_shape,x_shape])
    lun = x_shape - 1 
    PosMy,PosMx = np.where(Batch == 0)

    Pos_mx = PosMx.reshape(int(len(PosMx)/lun),lun)

    random_choosen = np.random.randint(lun) 
    posx = Pos_mx[:,random_choosen]

    posy = np.arange(len(posx))

    diff_Batch[posy[0:],posx[0:]] = 1 
    return diff_Batch

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
    o_h = 64
    o_w = 64
        
    s_h = np.array([o_h/16, o_h/8, o_h/4, o_h/2, o_h])
    s_w = np.array([o_w/16, o_w/8, o_w/4, o_w/2, o_w])
    
    
    s_h = s_h.astype(int)
    s_w = s_w.astype(int)
  
    with tf.variable_scope("generator_conv", reuse=Reuse):
        
       #x = inputs
        x = z
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
        x = conv_transpose(x, [batch_size, s_h[2], s_w[2], 32], "G_conv0",initializer=init,strides=[1,4,4,1])
        x = Batchnorm('Generator.BN1', [0,1,2], x)
        x = tf.nn.leaky_relu(x)
        
        
        #x = conv_transpose(x, [batch_size, s_h[4], s_w[4], 64], "G_conv1",initializer=init,strides=[1,4,4,1])
        #x = Batchnorm('Generator.BN2', [0,1,2], x)
        #x = tf.nn.leaky_relu(x)
        
        
        #x = conv_transpose(x, [batch_size, s_h[3], s_w[3], 32],"G_conv2",initializer=init,strides=[1,2,2,1])
        #x = Batchnorm('Generator.BN3', [0,1,2], x)
        #x = tf.nn.leaky_relu(x)
       
        
        x = conv_transpose(x, [batch_size, s_h[4], s_w[4], 3],"G_conv3",initializer=init,strides=[1,4,4,1])
    
        x = tf.tanh(x)
        
        x = tf.reshape(x,shape=[-1,4096*3])

        return x
  
    
def discriminator(x,c,Reuse=True,):
    with tf.variable_scope('discriminator',reuse=Reuse):
        keep_prob = 0.5 
        init = tf.contrib.layers.xavier_initializer() #☻canbiare 
        
                                                   
        x = gaussian_noise_layer(x)                                           
        x = tf.reshape(x, [-1,64,64,3])
        
        x = (conv2d(x, 3, 32, "D_conv0",initializer=init,strides = [1,4,4,1]))
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dropout(x, keep_prob)
        
        
        x = (conv2d(x, 32, 32, "D_conv1",initializer=init,strides = [1,4,4,1]))
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dropout(x, keep_prob)

        
        #x = (conv2d(x, 64, 64, "D_conv2",initializer=init,strides = [1,2,2,1]))
        #x = tf.nn.leaky_relu(x)
        #x = tf.layers.dropout(x, keep_prob)

        
        #x = (conv2d(x, 64,64, "D_conv3",initializer=init,strides = [1,2,2,1]))
        #x = tf.nn.leaky_relu(x)
        #x = tf.layers.dropout(x, keep_prob)
        
        #x = (conv2d(x, 64,128, "D_conv4",initializer=init,strides = [1,1,1,1]))
        #x = tf.nn.leaky_relu(x)
        #x = tf.layers.dropout(x, keep_prob)

                              
        #x = (conv2d(x, 128,128, "D_conv5",initializer=init,strides = [1,1,1,1]))
        #x = tf.nn.leaky_relu(x)
        #x = tf.layers.dropout(x, keep_prob)
        
        #x = (conv2d(x, 128,128, "D_conv6",initializer=init,strides = [1,1,1,1]))
        #x = tf.nn.leaky_relu(x)
        #x = tf.layers.dropout(x, keep_prob)
    
        
        x = tf.reshape(x,[-1,4*4*32])
        feature_match = x        
        print("x shape: ",x.shape)
        print("c shape: ",c.shape)
        
        x = tf.concat(values=[x, c],axis=1)
        
        hash_stream = tf.nn.sigmoid(tf.layers.dense(x, units=32,kernel_initializer=init,activation=None))

        #x = minibatch_discriminator(x)
        #x = tf.layers.dropout(x, keep_prob)
        classification = tf.layers.dense(x, units=10,kernel_initializer=init,activation=None)
        logits = tf.layers.dense(x,1,kernel_initializer=init, name='logits')
#                out = tf.layers.dense(x, units=10,kernel_initializer=init,activation=None)

        return logits,feature_match,classification,hash_stream


def Save_generator(G_vars):
    saver = tf.train.Saver(var_list=G_vars)
    saver.save(sess, 'my_generator_model.ckpt')
    print('salvo il generatore')
    
def Load_generator():
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_conv'))
    saver.restore(session, 'my_generator_model.ckpt')


def train(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step,
        D_extra_step,show_every=250, print_every=50, batch_size=128, num_epoch=10):

    
      #salva loss per tensorfboard 
    acc_step = 0 
    tf.summary.scalar("Discriminator loss", D_loss)
    tf.summary.scalar("Generator loss", G_loss)
    tf.summary.scalar("Accuracy",acc_step)  
    merged_summary_op = tf.summary.merge_all()
    logs_path = '/tmp/tensorflow_logs/example/'
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    
    N_EuroSatTrain_ex = 22950
    max_iter = int(N_EuroSatTrain_ex*num_epoch/batch_size)
    max_iter = 1
    print("max iter: ",max_iter)
    

    #for it_d in range(100):
    #    minibatch,minibatch_y = load_batch(batch_size=batch_size)

        #minibatch,minibatch_y = Get_batch(Eurosat_data,Eurosat_labels,it_d,batch_size)    
    #    _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch,y:minibatch_y})
    #    if(it_d % print_every == 0):
    #        print('Iter: {}, D: {:.4}'.format(it_d,D_loss_curr))
            
            
    #print("save model")
    #save_path = saver.save(sess, "/tmp/EuroSat_C_WGANGP_model.ckpt")
    #print("Model saved in path: %s" % save_path)
    
    '''
    init_variables()
    saver.restore(sess, "/tmp/EuroSat_TDiscriminator_model.ckpt")
    '''
    for it in range(max_iter):
        
       

        
        for _ in range(critich): 
            minibatch,minibatch_y = load_batch(batch_size=batch_size)
        
            #minibatch,minibatch_y = Get_batch(Eurosat_data,Eurosat_labels,it,batch_size)  
            Neg_Batch = Find_diff_batch(minibatch_y) 
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch,y:minibatch_y,Neg_Cond:Neg_Batch,D_t:1,T_t:0})
        
        _, G_loss_curr,summary = sess.run([G_train_step, G_loss,merged_summary_op], feed_dict={x: minibatch,y:minibatch_y,Neg_Cond:Neg_Batch,D_t:1,T_t:0})
        
     
        if(it % print_every == 0):
           # Eursat_data_test,Eurosat_labels_test
            _,_,Class,_ = discriminator(preprocess(x),tf.zeros_like(Eurosat_labels_test))
            
            target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4', 'class 5','class 6', 'class 7', 'class 8','class 9']
            lab = np.array(sess.run(tf.argmax(Eurosat_labels_test,1)))
            Class = tf.argmax(Class,1)
            cla = np.array(sess.run(Class,feed_dict={x:Eursat_data_test}))

            #cla = np.array(sess.run(tf.argmax(Class,1)))
            print("labels: ",lab)
            print("class: ",cla)
            
            print("report: ",classification_report(lab,cla, target_names=target_names))
            
             
            
            #con_mat = tf.confusion_matrix(Eurosat_labels_test,Class,10)
            correct = tf.equal(tf.argmax(Class,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            acc_step = accuracy.eval({x:Eursat_data_test, y:Eurosat_labels_test})
            print('Accuracy:',accuracy.eval({x:Eursat_data_test, y:Eurosat_labels_test}))
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
            
            #con_mat = tf.Variable()
            #A = sess.run(tf.variables_initializer([con_mat]))
            A,B = tf.metrics.mean_per_class_accuracy(tf.argmax(Eurosat_labels_test,1),tf.argmax(Class,1),10)
            tf.local_variables_initializer().run()
            print("shape A : ", A)
            print("shape B : ", B)
            print('Confusion Matrix: \n\n', sess.run(A))
            print('Confusion Matrix: ',sess.run(B))
        if(it % show_every == 0):
            
            
            plt.plot(x,y)
            plt.plot(x,z)
            plt.ylabel('GAN vs CNN')
            plt.xlabel('Number of labeled data')
            plt.show()
            
            samples = sess.run(G_sample,feed_dict={y:minibatch_y})
            
            samples = deprocess(samples)
            #save_images(samples[:16],it)
            
            
            fig = disp_images(samples[:16])
            plt.show()
            print()
        
            
    print('Final images')
    
    samples = sess.run(G_sample,feed_dict={y:minibatch_y})
    samples = deprocess(samples)
    #save_images(samples,int(it+1),last=True)
   
    
    _,_,Class,_ = discriminator(preprocess(x),tf.zeros_like(Eurosat_labels_test))

    
    print("immagini condizionate")
    for i in range(10):
        print("immagine n:",i)
        c = np.zeros(shape=[batch_size,y_dim])
        c[:,i] = 1
        c = tf.cast(c,tf.float32)
        c = sess.run(c)
        samples = sess.run(G_sample,feed_dict={y:c})
        samples = deprocess(samples)
        #save_images(samples[:8],i,last=False,cond=True,type_f_image=i)
        fig = disp_images(samples[:16])
        plt.show()
        
        
    print("save model")
    save_path = saver.save(sess, "/tmp/EuroSat_complete_model.ckpt")
    print("Model saved in path: %s" % save_path)
    
    '''
    init_variables()
    saver.restore(sess, "/tmp/EuroSat_complete_model.ckpt")
    '''
    return samples
    

    
y = tf.placeholder(tf.float32, shape=[None, y_dim])
h = tf.placeholder(tf.float32, shape=[None, y_dim])
x = tf.placeholder(tf.float32, [None,4096*3])
z = sample_noise(batch_size, noise_dim) #batch_size

def soft_noisy_loss(real_discriminator,fake_discriminator,BATCH_SIZE=128):
    # Train on soft labels (add noise to labels as well)
    noise_prop = 0.05 # Randomly flip 5% of labels
    
    # Prepare labels for real data
    true_labels = np.ones((BATCH_SIZE, 1)) - np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
    flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop*len(true_labels)))
    true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
    true_labels = tf.cast(true_labels,tf.float32)
    
    # Train discriminator on real data
    #real_discriminator = discriminator(images)
    d_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_discriminator, labels=true_labels))


    # Prepare labels for generated data
    gene_labels = np.zeros((BATCH_SIZE, 1)) + np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
    flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
    gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
    gene_labels = tf.cast(gene_labels,tf.float32)
    
    # Train discriminator on generated data
    #fake_discriminator = discriminator(generated_images)
    d_loss_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_discriminator, labels=gene_labels))

    d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)

    # Train generator
    #sample_noise(batch_size, dim,max_val=1,min_val=-1)
    
    g_loss = - tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_discriminator, labels=tf.zeros_like(fake_discriminator)))

    return d_loss,g_loss


def wgan_loss(logits_real, logits_fake, batch_size,x,G_sample,MODE):
    d_loss =- tf.reduce_mean(logits_real) + tf.reduce_mean(logits_fake)
    g_loss =- tf.reduce_mean(logits_fake)
    
    
    lam = 10
    LAMBDA_2 = 2.0
    
    eps = tf.random_uniform([batch_size,1], minval=0.0,maxval=1.0)
    x_h = eps*x+(1-eps)*G_sample
    
    with tf.variable_scope("", reuse=True) as scope:
        disc_real,f_disc_real,_,_ = discriminator(x_h)
        grad_d_x_h = tf.gradients(disc_real, x_h)
    
    grad_norm = tf.norm(grad_d_x_h[0], axis=-1, ord='euclidean')   #-1 per canali 
    grad_pen = tf.reduce_mean(tf.square(grad_norm-1))
    
    d_loss+=lam*grad_pen
    
    
    return d_loss, g_loss

def lossless_triplet_loss(y_true,anchor,positive,negative, N = 32, beta=32, epsilon=1e-8):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    N  --  The number of dimension 
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)    
    Returns:
    loss -- real number, value of the loss
    """
 
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),1)
    
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),1)
    #Non Linear Values  
    # -ln(-x/N+1)
    pos_dist = -tf.log(-tf.divide((pos_dist),beta)+1+epsilon)
    neg_dist = -tf.log(-tf.divide((N-neg_dist),beta)+1+epsilon)
    # compute loss
    loss = neg_dist + pos_dist    

    return loss
 
    
Neg_Cond = tf.placeholder(tf.float32, shape=[None, y_dim])

G_sample = generator_conv(z,y,Reuse=False)
G_sample_noisy = gaussian_noise_layer(G_sample)
with tf.variable_scope("") as scope:
    
    logits_real,feature_real,C_real,Hash_anchor = discriminator(preprocess(x),tf.zeros_like(y),Reuse=False) 
    logits_real,feature_real,C_real,_ = discriminator(preprocess(x),C_real) 

    
    scope.reuse_variables() # da controllare 
    logits_fake,feature_fake,C_fake,Hash_positive = discriminator(G_sample_noisy,tf.zeros_like(y))
    logits_fake,feature_fake,C_fake,_ = discriminator(G_sample_noisy,c=C_fake)
    
    
    Negative_fake_image = generator_conv(z,Neg_Cond)
    __,_,_,Hash_negative = discriminator(Negative_fake_image,tf.zeros_like(y)) 

    
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator_conv')

d_solver, g_solver = get_solvers()





#soft noisy loss 
d_loss,g_loss = soft_noisy_loss(logits_real,logits_fake)

# lossless triplet loss
triplet_loss = lossless_triplet_loss(y,Hash_anchor,Hash_positive,Hash_negative,N=32,beta=32,epsilon=1e-8)
triplet_loss = tf.reduce_mean(triplet_loss)

#classification_loss 
print('logist shape: ',np.shape(C_real))
s_loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=C_real,labels=y) )
d_loss += s_loss

g_loss += tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=C_fake,labels=y) )

#image matching loss 
recon_weight = tf.cast(1.0, tf.float32)
g_loss += tf.reduce_mean(huber_loss(tf.reshape(preprocess(x), [-1,64*64*3]),tf.reshape(G_sample, [-1,64*64*3])))*recon_weight

#feature matching
tmp1 = tf.reduce_mean(feature_real, axis = 0)
tmp2 = tf.reduce_mean(feature_fake, axis = 0)

G_L2 = tf.reduce_mean(tf.square(tmp1 - tmp2))

g_loss +=G_L2

d_train_step = d_solver.minimize(d_loss, var_list=D_vars)
g_train_step = g_solver.minimize(g_loss, var_list=G_vars)
d_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
g_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator_conv')




triplet_loss = tf.multiply(T_t,triplet_loss)
d_loss = tf.multiply(D_t,d_loss)

d_loss =  triplet_loss + d_loss 

with session() as sess:
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    samples = train(sess,g_train_step,g_loss,d_train_step,d_loss,g_extra_step,d_extra_step,batch_size=batch_size,
          num_epoch=40)
    
    Save_generator(G_vars)
   
    
# da testare

  