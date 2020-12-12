#cost function examples 


import tensorflow as tf 
import os 
import sys 
import numpy as np 

BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(BASE_DIR, 'Desktop\working_gan_thesis\EuroSat_GAN'))

import EuroSatArchive
from EuroSatArchive import load_batch



def Unsupervised_Hashing_cost_function(features,Hash_vector,alpha=1,batch_size=128,p=1,d=1,K_length=32):
   
    print("images shape: ",np.shape(features))
    print("Hash vector shape: ",np.shape(Hash_vector))
    
    Hash_vector_1 = Hash_vector[:64]
    Hash_vector_2 = Hash_vector[64:128]

    
    features_extended  = np.repeat(features[np.newaxis ,: ,: ], 128, axis=0)
    features_extended = np.reshape(features_extended,[128*128,32],'F')
    
    feature_subtraction = np.tile(features,128)
    feature_subtraction = np.reshape(feature_subtraction,[128*128,32]) # da sostituire con la lunghezza presa in interno 
    Delta_features = features_extended - feature_subtraction
    
    b_tilde =  2*Hash_vector - 1 
    
    b_tilde_extended  = np.repeat(b_tilde[np.newaxis ,: ,: ], 128, axis=0)
    b_tilde_extended = np.reshape(b_tilde_extended,[128*128,32],'F')
    
    b_tilde_mult = np.tile(b_tilde,128)
    b_tilde_mult = np.reshape(b_tilde_mult,[128*128,32]) # da sostituire con la lunghezza presa in interno 
    
    
    cross_mul = np.sum(np.multiply(b_tilde_extended,b_tilde_mult),axis=1)
    
    L1 = alpha * tf.reduce_mean(tf.abs(Hash_vector - 1))
    
    print("li shape: ",L1)
    
    #Rtest= sess.run(tf.reduce_mean(Hash_vector,axis = 0)) 
    print("hash vector : ",sess.run(tf.reduce_mean(Hash_vector,axis = 0)) )
    
    L2 = tf.nn.l2_loss((tf.reduce_mean(Hash_vector,axis = 0)) - Hash_vector)
    print("l2 shape: ",sess.run(L2))

    
    
    S = tf.exp(-tf.norm(Delta_features,axis=1)/ (p*d))
    b_tilde_1 = 2*Hash_vector_1 - 1 
    b_tilde_2 = 2*Hash_vector_2 - 1 
    
    
    
    Sim_term = (tf.multiply(tf.transpose(b_tilde_1),tf.transpose(b_tilde_2)) + K_length) / (2*K_length)
    
    L3 = tf.reduce_sum(tf.abs(S - Sim_term),axis=0)
    print("L3: ", sess.run(L3))
    return  L1 + L2 + L3 



features,labels = load_batch(batch_size = 128)
Hash_vector = features[:,:32]
Hash_vector = np.array(Hash_vector,dtype=float)
#Eurosat_data_Test = np.reshape(Eurosat_data_Test, [-1,64,64,3])


Loss_output = Unsupervised_Hashing_cost_function(features,Hash_vector)

sess = tf.Session() 
print("loss test : ",sess.run(Loss_output))



