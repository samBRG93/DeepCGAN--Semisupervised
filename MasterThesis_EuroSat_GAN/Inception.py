from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

from keras.callbacks import ModelCheckpoint

from keras.callbacks import TensorBoard

import os.path

from scipy.misc import imread, imresize

import tensorflow as tf 

from keras.layers import Embedding, Flatten, Input, merge

from keras.optimizers import Adam

import sys
import os 
import numpy as np 

BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(BASE_DIR, 'Desktop\working_gan_thesis\EuroSat_GAN'))

import EuroSatArchive
from EuroSatArchive import load_batch


from keras import optimizer 



def Get_dataset(Train=True):
    if Train==True:
        Archive_dim = 22950
    if Train==False:
        Archive_dim = 27000-22950
        
    Images,Labels = load_batch(Archive_dim,Train)
    return Images,Labels

def _get_triplet_mask(labels):

    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.



    A triplet (i, j, k) is valid if:

        - i, j, k are distinct

        - labels[i] == labels[j] and labels[i] != labels[k]



    Args:

        labels: tf.int32 `Tensor` with shape [batch_size]

    """

    # Check that i, j and k are distinct

    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)

    indices_not_equal = tf.logical_not(indices_equal)

    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)

    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)

    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)



    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)





    # Check if labels[i] == labels[j] and labels[i] != labels[k]

    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    i_equal_j = tf.expand_dims(label_equal, 2)

    i_equal_k = tf.expand_dims(label_equal, 1)



    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))



    # Combine the two masks

    mask = tf.logical_and(distinct_indices, valid_labels)



    return mask

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def batch_all_triplet_loss(labels,embeddings, margin=0.2, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    
    
    
    
    
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    
    mask = _get_triplet_mask(labels)    
    mask = tf.to_float(mask)    
    triplet_loss = tf.multiply(mask, triplet_loss)    

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets
 
    
def triplet_loss(x):
    anchor, positive, negative = x

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


nb_train_samples = 2500
nb_validation_samples = 800
 
    
#Eurosat_data_Train,Eurosat_labels_Train = Get_dataset()
#
#Eurosat_data_Validation = Eurosat_data_Train[:nb_validation_samples]
#Eurosat_labels_Validation = Eurosat_labels_Train[:nb_validation_samples]
#
#Eurosat_data_Test,Eurosat_labels_Test = Get_dataset(Train=False)
#
#Eurosat_data_Test = np.reshape(Eurosat_data_Test, [-1,64,64,3])
#Eurosat_data_Validation = np.reshape(Eurosat_data_Validation, [-1,64,64,3])


Eurosat_data_Train,Eurosat_labels_Train = load_batch(batch_size = 1000)

Eurosat_data_Validation = Eurosat_data_Train[:200]
Eurosat_labels_Validation = Eurosat_labels_Train[:200]

Eurosat_data_Test,Eurosat_labels_Test = load_batch(batch_size = 500,False)

Eurosat_data_Test = np.reshape(Eurosat_data_Test, [-1,64,64,3])
Eurosat_data_Validation = np.reshape(Eurosat_data_Validation, [-1,64,64,3])
Eurosat_data_Train = np.reshape(Eurosat_data_Train, [-1,64,64,3])

# create the base pre-trained model

base_model = InceptionV3(weights='imagenet', include_top=False)

# dimensions of our images.

#Inception input size

img_width, img_height = 299, 299



top_layers_checkpoint_path = 'cp.top.best.hdf5'

fine_tuned_checkpoint_path = 'cp.fine_tuned.best.hdf5'

new_extended_inception_weights = 'final_weights.hdf5'



top_epochs = 50

fit_epochs = 50



batch_size = 10



# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)
embeddings = Dense(32, activation='sigmoid')(x)


# and a logistic layer -- we have 2 classes
#predictions = Dense(2, activation='softmax')(x)


def base_loss(embeddings):
    def identity_loss(y_true, y_pred,):
        return batch_all_triplet_loss(y_true,embeddings)
    return identity_loss

# this is the model we will train
#
#loss = merge(
#       [labels,embeddings],
#       mode=batch_all_triplet_loss,
#       name='loss',
#       output_shape=(1, ))



model = Model(input=base_model.input, output=embeddings)



if os.path.exists(top_layers_checkpoint_path):

	model.load_weights(top_layers_checkpoint_path)

	print ("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False



# compile the model (should be done *after* setting layers to non-trainable)



# prepare data augmentation configuration
train_datagen = ImageDataGenerator(

    rescale=None,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



test_datagen = ImageDataGenerator(None)



train_generator = train_datagen.flow(

    Eurosat_data_Train,
    
    Eurosat_labels_Train,

    #target_size=(img_height, img_width),

    batch_size=batch_size

    #class_mode='categorical'
    )



validation_generator = test_datagen.flow(

    Eurosat_data_Validation,
    
    Eurosat_labels_Validation,

    #target_size=(img_height, img_width),

    batch_size=batch_size,

    #class_mode='categorical'
    )





#Save the model after every epoch.

mc_top = ModelCheckpoint(top_layers_checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)



#Save the TensorBoard logs.

tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)



# train the model on the new data for a few epochs

#model.fit_generator(...)



model.fit_generator(

    train_generator,

    samples_per_epoch=nb_train_samples // batch_size,

    nb_epoch=top_epochs,

    validation_data=validation_generator,

    nb_val_samples=nb_validation_samples // batch_size,

    callbacks=[mc_top, tb])



# at this point, the top layers are well trained and we can start fine-tuning

# convolutional layers from inception V3. We will freeze the bottom N layers

# and train the remaining top layers.



# let's visualize layer names and layer indices to see how many layers

# we should freeze:

for i, layer in enumerate(base_model.layers):

   print(i, layer.name)





#Save the model after every epoch.

mc_fit = ModelCheckpoint(fine_tuned_checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)





if os.path.exists(fine_tuned_checkpoint_path):

	model.load_weights(fine_tuned_checkpoint_path)

	print ("Checkpoint '" + fine_tuned_checkpoint_path + "' loaded.")



# we chose to train the top 2 inception blocks, i.e. we will freeze

# the first 172 layers and unfreeze the rest:

for layer in model.layers[:172]:

   layer.trainable = False

for layer in model.layers[172:]:

   layer.trainable = True



# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])



# we train our model again (this time fine-tuning the top 2 inception blocks

# alongside the top Dense layers

#model.fit_generator(...)

model.fit_generator(

    train_generator,

    samples_per_epoch=nb_train_samples // batch_size,

    nb_epoch=fit_epochs,

    validation_data=validation_generator,

    nb_val_samples=nb_validation_samples // batch_size,

    callbacks=[mc_fit, tb])


model.save_weights(new_extended_inception_weights)