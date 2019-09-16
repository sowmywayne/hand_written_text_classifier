from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

file = 'handwritten.csv'
data = pd.read_csv(file).astype('float32')

# rename is used to rename column 0 to label
data.rename(columns={'0':'label'}, inplace=True)
# feature and labels are changed
x = data.drop('label', axis = 1)
y = data['label']

# data is splited for training and testing
(x_train, x_test, y_train, y_test) = train_test_split(x, y)

# data cleaning
standard_scale = MinMaxScaler()
standard_scale.fit(x_train)
x_train = standard_scale.transform(x_train)
x_test = standard_scale.transform(x_test)

# reshaping the train and test data for traing the model
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# labeling the data for training
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# sequential model for training the data
cls = Sequential()
# conv2D layer with 32 filters and input size of (28, 28, 1)
cls.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
cls.add(MaxPooling2D(pool_size=(2, 2)))
# drop out the unwanted features
cls.add(Dropout(0.3))
# flatten the data
cls.add(Flatten())
# Dense layer with 128 filter and actiivation function of relu
cls.add(Dense(128, activation='relu'))
cls.add(Dense(len(y.unique()), activation='softmax'))
# summary of the layers and action functions
cls.summary()

# compiling the function with the layers which previously created
cls.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# training the model
cls.fit(x_train, y_train, epochs = 10, batch_size = 12)
# saving the trained model
cls.save('handWritten.h5')



