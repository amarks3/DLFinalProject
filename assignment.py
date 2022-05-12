import os
import sys
import argparse
import re
import tensorflow_datasets as tfds
import random
import cv2 as cv
from datetime import datetime
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import imag
from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, InputLayer, MaxPooling2D, Dropout, Conv2DTranspose,Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.activations import hard_sigmoid
from google.colab.patches import cv2_imshow
import h5py
from skimage.transform import resize
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint


def train_function(model, train,val, checkpoint_path, init_epoch):
    """ Training routine. Batch size 100 for 40 epochs"""
    print("training!")
    train = tf.data.Dataset.batch(train,100)
    val = tf.data.Dataset.batch(val,100)
    print(train)
    print(val)
    
    # Keras callbacks for training
    callback_list = [
        ModelCheckpoint(checkpoint_path,save_weights_only=True,
        monitor='val_iou',
        mode='max',
        save_best_only=True) 
    ]

    # Begin training
    
    model.fit(
        train,
        validation_data=val,
        epochs=40,
        batch_size=None,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )

def visualize_results(model, images, labels):
    """ visualizes the results version 1. """
    ims =list(tfds.as_numpy(images))
    lbs = list(tfds.as_numpy(labels))
    for i in range(len(ims)):
      im = tf.reshape(ims[i], (1,256,256,1))
      result = model.predict(im)
      result = result[0,:,:,0]*255
      print(i)
      cv2_imshow(result)
      print(" ")
      cv2_imshow(lbs[i]*255)
def test_function(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )

def test_model(model,images, labels):
    """ testing the model. Find the 10 best and 10 worst images and visualizes them with the 
    mask and the predictions."""
    ims =list(tfds.as_numpy(images))
    lbs = list(tfds.as_numpy(labels))
    acc_list =list()
    result_list =list()
    for i in range(len(ims)):
        im = ims[i]
        lb = lbs[i]
        im = tf.reshape(im, (1,256,256,1))
        result = model.predict(im)
        result = result[0,:,:,0]*255
        result_list.append(result)
        loss_numerator = tf.reduce_sum(lb*result)
        loss_denom = tf.reduce_sum(result)+tf.reduce_sum(lb)
        acc_list.append(loss_numerator/loss_denom)

    sorted_accs = np.argsort(acc_list)
    best = sorted_accs[:10]
    worst = sorted_accs[:-10]

    for i in best:
        fig = plt.figure(figsize=(10, 7))
        
        # setting values to rows and column variables
        rows = 1
        columns = 3

        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 1)
        
        # showing image
        plt.imshow(lbs[i])
        plt.axis('off')
        plt.title("Ground Truth")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)
        
        # showing image
        plt.imshow(result_list[i])
        plt.axis('off')
        plt.title(f"Prediction: Accuracy = {acc_list[i]}")
        plt.savefig(f"/content/drive/My Drive/best{i}.png")

    for i in worst:
        fig = plt.figure(figsize=(10, 7))
        
        # setting values to rows and column variables
        rows = 1
        columns = 3

        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 1)
        
        # showing image
        plt.imshow(lbs[i])
        plt.axis('off')
        plt.title("Mask")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)
        
        # showing image
        plt.imshow(result_list[i])
        plt.axis('off')
        plt.title(f"Prediction: accuracy = {acc_list[i]}")

        plt.savefig(f"/content/drive/My Drive/worst{i}.png")


def main():
    """ Main function. """

    # set up stuff for save the model
    time_now = datetime.now()
    
    # create the model and compile    
    model = make_model(256,256,1)
    model(tf.keras.Input(shape=(256, 256, 1)))
    checkpoint_path = "./checkpoints" + os.sep + \
        "your_model" + os.sep + "model" + os.sep
        
    # Print summary of model
    model.summary()
   
    model.compile(
        optimizer=Adam(learning_rate=.001),
        loss= loss_fn,
        metrics=[
        tf.keras.metrics.MeanIoU(num_classes =2 , name = 'iou'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.Precision(name='precision'),
        ]
        )
    # get the locally saved data
    d = h5py.File('/content/drive/My Drive/images2.h5', 'r')
    i = h5py.File('/content/drive/My Drive/labels2.h5', 'r')
    labels = np.array(d['X_train'][:])
    data = np.array(i['Y_train'][:])
    combined_inputs_labels = list(zip(data,labels))

    # shuffle data
    np.random.shuffle(combined_inputs_labels)
    inputs = tf.convert_to_tensor(list(zip(*combined_inputs_labels))[0])
    labels = tf.convert_to_tensor(list(zip(*combined_inputs_labels))[1])

    # shuffle data
    print(" standardizing data")
    meanX = tf.math.reduce_mean(inputs)
    stdX = tf.math.reduce_std(inputs)
    trainX = (inputs-meanX)/stdX
    print(meanX, stdX)

    # create the datasets
    with h5py.File('test_ims.h5', 'w') as hf:
        hf.create_dataset("Y_train", data=trainX[3000:3100],dtype=np.float32)
    with h5py.File('test_lbs.h5', 'w') as hf:
        hf.create_dataset("X_train", data=labels[3000:3100],dtype=np.float32)

    train_X = tf.data.Dataset.from_tensor_slices(trainX[:3000])    
    val_X = tf.data.Dataset.from_tensor_slices(trainX[5500:])
    test_X = tf.data.Dataset.from_tensor_slices(trainX[3000:3100])

    train_y = tf.data.Dataset.from_tensor_slices(labels[:3000])
    val_y = tf.data.Dataset.from_tensor_slices(labels[5500:])
    test_y = tf.data.Dataset.from_tensor_slices(labels[3000:3100])

    train = tf.data.Dataset.zip((train_X, train_y))
    val = tf.data.Dataset.zip((val_X, val_y)) 

    # train the model
    train_function(model, train, val,checkpoint_path, 0)

    # save the model
    model.save("/content/drive/My Drive/my_h5_model.h5")

    # test the model
    test_model(model,test_X,test_y)

    #visualize_results(model, val_X, val_y)

# Make arguments global
def make_model(imgHeight, imgWidth,imgChannel):
    """ Model following a traditional unet, starting with 16 filtersd and going up. 
        Using leaky relu as suggested by firenet with the same sloped of line
    """
    inputs = Input((imgHeight, imgWidth,1))
    #s = Lambda(lambda x: x / 255)(inputs)
    
    c1 = Conv2D(4, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    c1 = LeakyReLU(.1)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(4, (3, 3),  kernel_initializer='he_normal', padding='same')(c1)
    c1 = LeakyReLU(.1)(c1)

    c1 = BatchNormalization()(c1)

    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(8, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = LeakyReLU(.1)(c2)

    c2 = BatchNormalization()(c2)
    c2 = Conv2D(8, (3, 3), kernel_initializer='he_normal', padding='same')(c2)
    c2 = LeakyReLU(.1)(c2)


    c2 = BatchNormalization()(c2)

    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = LeakyReLU(.1)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
    c3 = LeakyReLU(.1)(c3)

    c3 = BatchNormalization()(c3)

    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = LeakyReLU(.1)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(32, (3, 3),  kernel_initializer='he_normal', padding='same')(c4)
    c4 = LeakyReLU(.1)(c4)

    c4 = BatchNormalization()(c4)

    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
    c5 = LeakyReLU(.1)(c5)

    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
    c5 = LeakyReLU(.1)(c5)


    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = LeakyReLU(.1)(c6)

    c6 = BatchNormalization()(c6)
    c6 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = LeakyReLU(.1)(c6)

    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(16, (3, 3),  kernel_initializer='he_normal', padding='same')(u7)
    c7 = LeakyReLU(.1)(c7)

    c7 = BatchNormalization()(c7)
    c7 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
    c7 = LeakyReLU(.1)(c7)

    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(8, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
    c8 = LeakyReLU(.1)(c8)

    c8 = BatchNormalization()(c8)
    c8 = Conv2D(8, (3, 3), kernel_initializer='he_normal', padding='same')(c8)
    c8 = LeakyReLU(.1)(c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(4, (3, 3),  kernel_initializer='he_normal', padding='same')(u9)
    c9 = LeakyReLU(.1)(c9)

    c9 = BatchNormalization()(c9)
    c9 = Conv2D(4, (3, 3), kernel_initializer='he_normal', padding='same')(c9)
    c9 = LeakyReLU(.1)(c9)

    c9 = BatchNormalization()(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def loss_fn(mask, predictions):
    """ Loss function for the model. Folowing the soft dice score"""
    loss_numerator = 2 * tf.reduce_sum(mask*predictions)+1
    loss_denom = tf.reduce_sum(predictions)+tf.reduce_sum(mask)+1
    return 1- (loss_numerator/loss_denom)

main()
