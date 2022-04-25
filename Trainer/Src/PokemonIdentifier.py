# %% [markdown]
# # Pokemon Identifier

# %% [markdown]
# This is going to train a pokemon identifier that will be trained on several data sources

# %%
import os
import sys
import string
import datetime
import shutil
import argparse
import random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.preprocessing.image as preprocessing 
from tensorflow.data import AUTOTUNE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
import tensorflow_addons as tfa
from imutils import paths #used to get the paths of all images in a dir

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# ## Global Values

# %%
clearLogs = False
strNow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
ENV = "ubuntuLocal"

#path to dataset directory
ENV_LOG_DIR = f"../Logs/{ENV}/"
SESSION_LOG_DIR = f"../Logs/{ENV}/{strNow}/"
CORE_DATASET = '../Datasets/Main/Images/'
TRAIN_DIR = '../Datasets/Main/Images/Train/'
VALIDATION_DIR = '../Datasets/Main/Images/Validation/'
# SAVE_DIR = os.path.join(SESSION_LOG_DIR, "Saves")
# CHECKPOINT_DIR = os.path.join(SAVE_DIR, "Checkpoints")
# FINAL_SAVE_DIR = os.path.join(SAVE_DIR, "Final")
if os.path.isdir('../Logs') is False:
    os.mkdir('../Logs')
if os.path.isdir(ENV_LOG_DIR) is False:
    os.mkdir(ENV_LOG_DIR)
if os.path.isdir(SESSION_LOG_DIR) is False: 
    os.mkdir(SESSION_LOG_DIR)

numCategories = len(os.listdir(TRAIN_DIR))

#amount of time to allot for training
trainTimeLimit = 0

#percentage of core dataset to use for training and testing
TRAIN_SPLIT = .9 

if clearLogs is True and os.path.isdir('../Logs') :
    shutil.rmtree('../Logs')
    
if os.path.isdir(CORE_DATASET) is False:
    print('DATASET NOT FOUND') 
    
if os.path.isdir(TRAIN_DIR) is False:
    print('TRAIN SET NOT FOUND')
                 
if os.path.isdir(VALIDATION_DIR) is False:
    print('VALIDATION SET NOT FOUND')

BATCH_SIZE = 512

NUM_EPOCHS = 10

IMAGE_HEIGHT = 70
IMAGE_WIDTH = 70
#normalization value that will be used for color channels
IMAGE_NORM_COLOR = 255 

# %% [markdown]
# ### Data Pipeline Params

# %%
PIPE_USE_RAND_ZOOM = True
PIPE_RAND_ZOOM_AMT = (-0.5, 0.5)

# %% [markdown]
# ### Training Params

# %%
NUM_NODES_IN_CONV = [128]
NUM_LAYERS_CONV = [3]
CONV_KERNEL_SIZE = (4,4)
REGULARIZER_USE = True
REGULARIZER_LEARNING_RATE = 0.01
USE_BATCH_NORMS = True

# %% [markdown]
# ### Take command line arguments if any

# %%
if (len(sys.argv) > 0):
    print(sys.argv)
    listArgs = sys.argv
    for arg in listArgs: 
        splitArg = arg.split('=')
        if splitArg[0] == "timeLimit": 
            timeSplit = splitArg[1].split('.')
            if (len(timeSplit) == 3):
                hours = int(timeSplit[0])
                minutes = int(timeSplit[1])
                seconds = int(timeSplit[2])
                trainTimeLimit = (60*60*hours)+(60*minutes)+seconds
                print(f'time limit set to- {trainTimeLimit} seconds') 
        elif splitArg[0] == "epochs":
            numEpochs = int(splitArg[1])

# %% [markdown]
# ## Train

# %% [markdown]
# ### Create Datasets with TF.Data

# %%
# iamgePaths = list(paths.list_images(CORE_DATASET))
# random.seed(32)
# random.shuffle(imagePaths)

# #generate training and testing split 
# i = int(len(imagePaths) * 

# datasets = [
#     ("training", trainPaths, TRAIN_DIR),
#     ("validation", validationPaths, VALIDATION_DIR)
# ]

# %%
def loadImages(imagePath):
    #encode the image
    # tf.print(imagePath)
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_png(image, channels=4)
    image = tf.image.resize_with_pad(image, IMAGE_HEIGHT, IMAGE_WIDTH)
    image = tf.cast(image, tf.float32)
    
    #encode the label for the image
    labelParts = tf.strings.split(imagePath, os.sep)
    oneHot = labelParts[-2] == classNames 
    return (image, tf.argmax(oneHot))

# %%
seqAugment = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(scale=1.0/255),
    tf.keras.layers.RandomZoom(
        height_factor=PIPE_RAND_ZOOM_AMT, #zoom in by random ammount from +20% to +30%
        width_factor=PIPE_RAND_ZOOM_AMT, 
        fill_mode='constant', 
        fill_value=0)
])

# %%
trainPaths = list(paths.list_images(TRAIN_DIR))
valPaths = list(paths.list_images(VALIDATION_DIR))
trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths] #gather labels from dirs 
classNames = np.array(sorted(trainLabels))
classNames = np.unique(classNames)

print(len(trainPaths))

#layer used to normalize colors in datasets 
normalizationLayer = tf.keras.layers.Rescaling(1./255)

#define pipelines 
#training dataset 
trainDS = tf.data.Dataset.from_tensor_slices(trainPaths)
trainDS = (trainDS
           .shuffle(len(trainPaths)) #shuffle all the images 
           .map(loadImages, num_parallel_calls=AUTOTUNE) #read images from disk 
           .map(lambda x, y: (seqAugment(x), y), num_parallel_calls=AUTOTUNE)
           .batch(BATCH_SIZE) #batch size
           .prefetch(AUTOTUNE)
          )

# trainDS = trainDS.map(lambda x, y: (normalizationLayer(x), y))
#validation dataset
valDS = tf.data.Dataset.from_tensor_slices(valPaths)
valDS = (valDS
         .map(loadImages, num_parallel_calls=AUTOTUNE)
         .batch(BATCH_SIZE)
         .prefetch(AUTOTUNE))
# valDS = valDS.map(lambda x, y: (normalizationLayer(x), y))

# %%
# print(trainPaths[15061])
for images, labels in trainDS.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow((images[i].numpy()*255).astype("uint8"))
        plt.title(classNames[labels[i]])
        plt.axis("off")
    plt.show()

# %% [markdown]
# ### Create Datagenerators

# %% [markdown]
# ### Define Model

# %%
for numConvLayers in NUM_LAYERS_CONV:
    for convNodes in NUM_NODES_IN_CONV:
        for convKernelSize in CONV_KERNEL_SIZE:
            local_container_dir = os.path.join(SESSION_LOG_DIR, f"cl{numConvLayers}.cn{convNodes}.ckern{convKernelSize}")
            print(local_container_dir)
            local_tensorlogs_dir = os.path.join(local_container_dir, 'fit')
            local_save_dir = os.path.join(local_container_dir, 'saves')
            local_checkpoint_dir = os.path.join(local_save_dir, 'checkpoints/')
            local_finalsave_dir = os.path.join(local_save_dir, 'final/')

            if os.path.isdir(local_container_dir) is False: 
                os.mkdir(local_container_dir)
            if os.path.isdir(local_save_dir) is False:
                os.mkdir(local_save_dir)
            if os.path.isdir(local_checkpoint_dir) is False:
                os.mkdir(local_checkpoint_dir)
            if os.path.isdir(local_finalsave_dir) is False: 
                os.mkdir(local_finalsave_dir)

            #write summary to file 
            infoFile = os.path.join(local_container_dir, "into.txt")
            with open(infoFile, 'w') as file: 
                file.write(f"Number of convolution layers: {numConvLayers} \r")
                file.write(f"Number of nodes per convolution layer: {convNodes} \r")
                file.write(f"Input size expected: {IMAGE_WIDTH}, {IMAGE_HEIGHT}\r")
                file.write(f"Epochs: {NUM_EPOCHS}\r")
                file.write(f"Regularizers in use? {REGULARIZER_USE}\r")
                file.write(f"Regularizer learning rate: 0.01\r")
                file.write(f"Conv kernel size: {convKernelSize}\r")
                file.close()

            #cleanup from last round 
            tf.keras.backend.clear_session()

            #define the model 
            model = tf.keras.models.Sequential()
            for i in range(numConvLayers):
                if REGULARIZER_USE is True:
                    model.add(tf.keras.layers.Conv2D(int(convNodes),
                                                        convKernelSize,
                                                        activation='relu',
                                                        kernel_regularizer=regularizers.l2(REGULARIZER_LEARNING_RATE)))
                else:
                    model.add(tf.keras.layers.Conv2D(int(convNodes), convKernelSize, activation='relu'))
                
                #add additional properties to conv layers 
                if USE_BATCH_NORMS is True:
                    model.add(tf.keras.layers.BatchNormalization())

                model.add(tf.keras.layers.MaxPool2D((2,2)))

            #flatten out 
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(512, activation='relu'))
            model.add(tf.keras.layers.Dense(len(classNames), activation='softmax'))
            
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=RMSprop(learning_rate=1e-4),
                    metrics=['accuracy'])
            
            #create callbacks as necessary
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=local_checkpoint_dir, 
                                                                    save_weights_only=True, 
                                                                    verbose=1)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=local_tensorlogs_dir,                        
                                                                histogram_freq=1, 
                                                                write_graph=True, 
                                                                write_images=False, 
                                                                embeddings_freq=1, 
                                                                profile_batch='1,5')
                
            time_stopping_callback = None
            history = None
            
            if (trainTimeLimit != 0):
                #shorten the time limit to allow for post training data processing
                trainTimeLimit = trainTimeLimit - (60*5) 
                time_stopping_callback = tfa.callbacks.TimeStopping(seconds=trainTimeLimit, verbose=1)
                history = model.fit(trainDS, 
                                    epochs=NUM_EPOCHS, 
                                    validation_data=valDS,
                                    callbacks=[tensorboard_callback, checkpoint_callback, time_stopping_callback])
            else:
                #no time limit callback
                history = model.fit(trainDS, 
                            epochs=NUM_EPOCHS, 
                            validation_data=valDS,
                            callbacks=[tensorboard_callback, checkpoint_callback])

            #save model and record completion in info file
            model.save(local_finalsave_dir)
            with open(infoFile, 'a') as file: 
                file.write('Training Complete')
                file.close()

# %% [markdown]
# #### Zip Logs For Download

# %%
shutil.make_archive('Logs', 'zip', '../Logs')


