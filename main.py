import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import math
import cv2
import datetime
import platform
import os

from sklearn.model_selection import train_test_split
from PIL import Image


# def get_number_from_txt(txt_file_path):
#     # Open the TXT file and read the first two characters
#     with open(txt_file_path, 'r') as file:
#         content = file.read(2)
#         # Convert the first two characters to a number
#         try:
#             number = int(content)
#             # Ensure the number is in the range of 1 to 99
#             return max(1, min(99, number))
#         except ValueError:
#             return None
#
#
# def convertTestToCSV (imgDir):
#     for img in os.listdir(imgDir):
#         _, file_extension = os.path.splitext(img)
#         if not file_extension.lower() == '.png':
#             print("Skipping non-PNG file: {img}")
#             continue
#         img_array = cv2.imread(os.path.join(imgDir, img), cv2.IMREAD_GRAYSCALE)
#         img_array = cv2.resize(img_array, (28, 28))
#         img_array = img_array.flatten()
#         img_array = img_array.reshape(-1, 1).T
#         with open('content/csv/test.csv', 'ab') as f:
#             np.savetxt(f, img_array, delimiter=",")
#
#
# def convertTrainToCSV (imgDir):
#     for img in os.listdir(imgDir):
#         _, file_extension = os.path.splitext(img)
#         if not file_extension.lower() == '.png':
#             print("Skipping non-PNG file: {img}")
#             continue
#         base_name = os.path.splitext(os.path.basename(img))[0]
#         number = get_number_from_txt(imgDir + '/' + base_name + '.txt')
#
#         img_array = cv2.imread(os.path.join(imgDir, img), cv2.IMREAD_GRAYSCALE)
#         img_array = cv2.resize(img_array, (28, 28))
#         img_array = img_array.flatten()
#         img_array = img_array.reshape(-1, 1).T
#         img_array = np.insert(img_array, 0, number)
#         with open('content/csv/train.csv', 'ab') as f:
#             np.savetxt(f, img_array, delimiter=",")
#
#
# trainDir = 'content/images/train'
# testDir = 'content/images/test'
#
# convertTestToCSV(testDir)
# convertTrainToCSV(trainDir)

train = pd.read_csv('content/csv/train.csv')
test = pd.read_csv('content/csv/test.csv')

X = train.iloc[:, 1:785]
y = train.iloc[:, 0]
X_test = test.iloc[:, 0:784]

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2,random_state = 1212)

x_train_re = X_train.to_numpy().reshape(33600, 28, 28)
y_train_re = y_train.values
x_validation_re = X_validation.to_numpy().reshape(8400, 28, 28)
y_validation_re = y_validation.values
x_test_re = test.to_numpy().reshape(28000, 28, 28)

(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train_re.shape
IMAGE_CHANNELS = 1

x_train_with_chanels = x_train_re.reshape(
    x_train_re.shape[0],
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS
)
x_validation_with_chanels = x_validation_re.reshape(
    x_validation_re.shape[0],
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS
)
x_test_with_chanels = x_test_re.reshape(
    x_test_re.shape[0],
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS
)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Convolution2D(
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
    kernel_size=5,
    filters=8,
    strides=1,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.keras.initializers.VarianceScaling()
))
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
))
model.add(tf.keras.layers.Convolution2D(
    kernel_size=5,
    filters=16,
    strides=1,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.keras.initializers.VarianceScaling()
))
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(
    units=128,
    activation=tf.keras.activations.relu
));
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(
    units=10,
    activation=tf.keras.activations.softmax,
    kernel_initializer=tf.keras.initializers.VarianceScaling()
))

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=adam_optimizer,
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

training_history = model.fit(
    x_train_normalized,
    y_train_re,
    epochs=100,
    validation_data=(x_validation_normalized, y_validation_re)
)

x_train_normalized = x_train_with_chanels / 255
x_validation_normalized = x_validation_with_chanels / 255
x_test_normalized = x_test_with_chanels / 255

plt.xlabel('Número de Épocas')
plt.ylabel('Acurácia')
plt.plot(training_history.history['loss'], label='Dataset Treinamento')
plt.plot(training_history.history['val_loss'], label='Dataset Validação')
plt.legend()

plt.xlabel('Número de Épocas')
plt.ylabel('Acurácia')
plt.plot(training_history.history['accuracy'], label='Dataset Treinamento')
plt.plot(training_history.history['val_accuracy'], label='Dataset Valicação')
plt.legend()