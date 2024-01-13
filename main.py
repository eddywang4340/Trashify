import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import model_from_json

# paper plastic glass aluminum
data = []
labels = []

# creating dataset for paper images
papers = os.listdir("ML-trash-images/paper")
for paper in papers:
    imag = cv2.imread("ML-trash-images/paper"+paper)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

# creating dataset for plastic images
plastics = os.listdir("ML-trash-images/plastic")
for plastic in plastics:
    imag = cv2.imread("ML-trash-images/plastic"+plastic)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)

# creating dataset for glass images
glasses = os.listdir("ML-trash-images/glass")
for glass in glasses:
    imag = cv2.imread("ML-trash-images/glass"+glass)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(2)

# creating dataset for aluminum images
aluminums = os.listdir("ML-trash-images/aluminum")
for aluminum in aluminums:
    imag = cv2.imread("ML-trash-images/aluminum"+aluminum)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(3)

# converting and saving dataset into numpy array
trash = np.array(data)
labels = np.array(labels)
np.save("trash",trash)
np.save("labels",labels)

# loading dataset
trash = np.load("trash.npy")
labels = np.load("labels.npy")

# shuffling dataset
s = np.arange(trash.shape[0])
np.random.shuffle(s)
trash = trash[s]
labels = labels[s]

types_trash = len(np.unique(labels))
data_length = len(trash)

# creating train and test datasets
x_train = trash[(int)(0.1*data_length):]
x_test = trash[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

train_length = len(x_train)
test_length = len(x_test)

# creating train and test label dataset
y_train = labels[(int)(0.1*data_length):]
y_test = labels[:(int)(0.1*data_length)]

# one-hot encoding label dataset
y_train = keras.utils.to_categorical(y_train, types_trash)
y_test = keras.utils.to_categorical(y_test, types_trash)
