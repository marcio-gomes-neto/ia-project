import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import math
import datetime
import platform
import os

from sklearn.model_selection import train_test_split
from PIL import Image


def get_number_from_txt(txt_file_path):
    # Open the TXT file and read the first two characters
    with open(txt_file_path, 'r') as file:
        content = file.read(2)
        # Convert the first two characters to a number
        try:
            number = int(content)
            # Ensure the number is in the range of 1 to 99
            return max(1, min(99, number))
        except ValueError:
            return None


def resize_image(input_path, output_path, target_size=(28, 28)):
    img = Image.open(input_path)
    img_resized = img.resize(target_size)
    img_resized.save(output_path)


def image_to_csv_train(image_folder, csv_output_file):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    df = pd.DataFrame(columns=['label'] + [f'pixel_{i}' for i in range(28 * 28)])

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        base_name = os.path.splitext(os.path.basename(image_file))[0]
        txt_file_path = os.path.join(directory, f"{base_name}.txt")
        number = get_number_from_txt(txt_file_path)

        resized_image_path = os.path.join(image_folder, 'resized', image_file)
        resize_image(image_path, resized_image_path)
        resized_img = Image.open(resized_image_path)
        resized_img = resized_img.convert('L')
        pixel_values = list(resized_img.getdata())
        row_data = [number] + pixel_values
        df = df.append(pd.Series(row_data, index=df.columns), ignore_index=True)

    df.to_csv(csv_output_file, index=False)


def image_to_csv_test(image_folder, csv_output_file):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    df = pd.DataFrame(columns=[f'pixel_{i}' for i in range(28 * 28)])

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        resized_image_path = os.path.join(image_folder, 'resized', image_file)
        resize_image(image_path, resized_image_path)
        resized_img = Image.open(resized_image_path)
        resized_img = resized_img.convert('L')
        pixel_values = list(resized_img.getdata())
        df = df.append(pd.Series(pixel_values, index=df.columns), ignore_index=True)

    df.to_csv(csv_output_file, index=False)


image_folder_path_test = '/content/images/test/'
image_folder_path_train = '/content/images/train/'
csv_output_file_test = '/content/csv/test.csv'
csv_output_file_train = '/content/csv/train.csv'

image_to_csv_test(image_folder_path_test, csv_output_file_test)
image_to_csv_train(image_folder_path_train, csv_output_file_train)

train = pd.read_csv('/content/csv/train.csv')
test = pd.read_csv('/content/csv/test.csv')

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