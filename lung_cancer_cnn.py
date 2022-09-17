from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d
import tflearn
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib inline
import seaborn as sns
import cv2
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121

import keras

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.initializers import random_uniform, glorot_uniform, identity, constant
from matplotlib.pyplot import imshow
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import np_utils
from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.keras.datasets import cifar10
import numpy as np
from keras import optimizers, losses
from keras.layers import *
from keras.backend import int_shape
import pydicom
from matplotlib import pyplot, cm
import glob

st = time.time()
os.listdir('/home/bfarhadi/miniconda3')

for d in os.listdir('/gpfs/project/6066240/bfarhadi/sample_images/stage1/'):
    print("Patient '{}' has {} scans".format(
        d, len(os.listdir('/gpfs/project/6066240/bfarhadi/sample_images/stage1/' + d))))
print('----')
print('Total patients {} Total DCM files {}'.format(len(os.listdir('/gpfs/project/6066240/bfarhadi/sample_images/stage1/')),
      len(glob.glob('/gpfs/project/6066240/bfarhadi/sample_images/stage1/*/*.dcm'))))


sizes = [os.path.getsize(dcm)/1000000 for dcm in glob.glob(
    '/gpfs/project/6066240/bfarhadi/sample_images/stage1/*/*.dcm')]
print('DCM file sizes: min {:.3}MB max {:.3}MB avg {:.3}MB std {:.3}MB'.format(np.min(sizes),
                                                       np.max(sizes), np.mean(sizes), np.std(sizes)))


df = pd.read_csv('/home/bfarhadi/miniconda3/sample_images/stage1_labels.csv')
df.head(5)
print('----')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


print('chunk')


def mean(l):
  return sum(l)/len(l)


print('mean')


num_patients = 26
# epochs =  20

patients = os.listdir('/gpfs/project/6066240/bfarhadi/sample_images/stage1')
patients.sort()
labels_df = pd.read_csv(
    '/home/bfarhadi/miniconda3/sample_images/stage1_labels.csv')

print('Patient')

trainingData = complete_data[:-1*int(num_patients*.80)]
validationData = complete_data[-1*int(num_patients*.80):]
# print(train_data.shape)

all_slices = np.array(all_slices)
all_labels = np.array(all_labels)

x_train = all_slices[:-1*int(num_patients*.80)]
y_train = all_labels[:-1*int(num_patients*.80)]
x_val = all_slices[-1*int(num_patients*.80):]
y_val = all_labels[-1*int(num_patients*.80):]

print('Patient')


tf.compat.v1.disable_v2_behavior()

# imageData = np.load('complete_data-px512-py512-sc26.npy')

# imageData = np.load('/content/complete_data-px512-py512-sc26.npy',allow_pickle=True)
# trainingData = imageData[0:18]
# validationData = imageData[-6:-3]

devices = tf.config.experimental.list_physical_devices('GPU')
devices=["/GPU:0" , "/GPU:1" , "/GPU:2" , "/GPU:3" ] # As many as you want
mirrored_strategy = tf.distribute.MirroredStrategy(devices)

x = tf.compat.v1.placeholder('float')
y = tf.compat.v1.placeholder('float')
size = 128
keep_rate = 0.8
NoSlices = 10


def convolution3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpooling3d(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def cnn(x):
    x = tf.reshape(x, shape=[-1, size, size, NoSlices,  1])
    convolution1 = tf.nn.relu(
        convolution3d(x, tf.Variable(tf.random.normal([3, 3, 3, 1, 32]))) + tf.Variable(tf.random.normal([32])))
    convolution1 = maxpooling3d(convolution1)
    convolution2 = tf.nn.relu(
        convolution3d(convolution1, tf.Variable(tf.random.normal([3, 3, 3, 32, 64]))) + tf.Variable(
            tf.random.normal([64])))
    convolution2 = maxpooling3d(convolution2)
    convolution3 = tf.nn.relu(
        convolution3d(convolution2, tf.Variable(tf.random.normal([3, 3, 3, 64, 128]))) + tf.Variable(
            tf.random.normal([128])))
    convolution3 = maxpooling3d(convolution3)
    convolution4 = tf.nn.relu(
        convolution3d(convolution3, tf.Variable(tf.random.normal([3, 3, 3, 128, 256]))) + tf.Variable(
            tf.random.normal([256])))
    convolution4 = maxpooling3d(convolution4)
    convolution5 = tf.nn.relu(
        convolution3d(convolution4, tf.Variable(tf.random.normal([3, 3, 3, 256, 512]))) + tf.Variable(
            tf.random.normal([512])))
    convolution5 = maxpooling3d(convolution4)
    fullyconnected = tf.reshape(convolution5, [-1, 1024])
    fullyconnected = tf.nn.relu(
        tf.matmul(fullyconnected, tf.Variable(tf.random.normal([1024, 1024]))) + tf.Variable(tf.random.normal([1024])))
    fullyconnected = tf.nn.dropout(fullyconnected, rate=1 - (keep_rate))
    output = tf.matmul(fullyconnected, tf.Variable(
        tf.random.normal([1024, 2]))) + tf.Variable(tf.random.normal([2]))
    return output


print('----------------------convolution3d----------------------------')

def network(x):
    prediction = cnn(x)
    cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=tf.stop_gradient(y)))
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=1e-3).minimize(cost)
    correct = tf.equal(tf.argmax(input=prediction, axis=1), tf.cast(y,'int64'))

    epochs = 10
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        tf.compat.v1.summary.FileWriterCache.clear()

        writer1 = tf.compat.v1.summary.FileWriter('./logs/acc/validation',session.graph)
        accuracy = tf.reduce_mean(input_tensor=tf.cast(correct, 'float'))
        accuracy_scalar = tf.compat.v1.summary.scalar("Accuracy_validation",accuracy)

        writer2 = tf.compat.v1.summary.FileWriter('./logs/loss',session.graph)
        loss = tf.Variable(0, dtype=tf.float32)
        loss_scalar = tf.compat.v1.summary.scalar("Loss",loss)

        writer3 = tf.compat.v1.summary.FileWriter('./logs/acc/train',session.graph)
        accuracy_tr = tf.Variable(0, dtype=tf.float32)
        acc_train = tf.compat.v1.summary.scalar("Accuracy_train",accuracy_tr)

        tf.compat.v1.summary.merge_all()
        tf.compat.v1.summary.merge([accuracy_scalar,acc_train]) 
        
        for epoch in range(epochs):
            epoch_loss = 0
            for data in trainingData:
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = session.run([optimizer, cost],
                                       feed_dict={x: X, y: Y})
                    epoch_loss += c
                except Exception as e:
                    pass

            summ1 = session.run(accuracy_scalar, feed_dict={x: [x_val], y: [y_val]})
            writer1.add_summary(summ1, epoch)
            writer1.flush()
            
            session.run(loss.assign(epoch_loss))
            writer2.add_summary(session.run(loss_scalar), epoch)
            writer2.flush()
            
            summ3 = np.mean([accuracy.eval({x: [d[0]], y: [d[1]]}) for d in trainingData])
            session.run(accuracy_tr.assign(summ3))
            writer3.add_summary(session.run(acc_train),epoch)
            writer3.flush()
            
            print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
            print('Accuracy:', accuracy.eval({x: [x_val], y: [y_val]}))
        
        print('Final Accuracy:', accuracy.eval({x: [x_val], y: [y_val]}))
        writer1.close()
        writer2.close()
        writer3.close() 

        patients = []
        actual = []
        predicted = []

        finalprediction = tf.argmax(input=prediction, axis=1)
        actualprediction = tf.cast(y,'int64')
        for i in range(len(validationData)):
            patients.append(validationData[i][1])
        for i in finalprediction.eval({x: [x_val], y: [y_val]}):
            if(i==1):
                predicted.append("Cancer")
            else:
                predicted.append("No Cancer")
        for i in actualprediction.eval({x: [x_val], y: [y_val]}):
            if(i[0]==1):
                actual.append("Cancer")
            else:
                actual.append("No Cancer")
        for i in range(len(patients)):
            print(i)

  #      from sklearn.metrics import confusion_matrix
  #      y_actual = pd.Series(
  #          (actualprediction.eval({x: [x_val], y: [y_val]})),
  #          name='Actual')
  #      y_predicted = pd.Series(
  #          (finalprediction.eval({x: [x_val], y: [y_val]})),
  #          name='Predicted')
  #      df_confusion = pd.crosstab(y_actual, y_predicted)
  #      print(df_confusion)

   #     def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):\
            
    #        plt.matshow(df_confusion, cmap=cmap)  # imshow  
            # plt.title(title)
     #       plt.colorbar()
      #      tick_marks = np.arange(len(df_confusion.columns))
      #      plt.xticks(tick_marks, df_confusion.columns, rotation=45)
      #      plt.yticks(tick_marks, df_confusion.index)
            # plt.tight_layout()
      #      plt.ylabel(df_confusion.index.name)
      #      plt.xlabel(df_confusion.columns.name)
      #      plt.show()
        #plot_confusion_matrix(df_confusion)
        # print(y_true,y_pred)
        # print(confusion_matrix(y_true, y_pred))
        # print(actualprediction.eval({x:[i[0] for i in validationData], y:[i[1] for i in validationData]}))
        # print(finalprediction.eval({x:[i[0] for i in validationData], y:[i[1] for i in validationData]}))
network(x)

ed = time.time()

print(f'Duration: {ed-st:5f}s')

time.sleep(1000)
