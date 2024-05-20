import tensorflow as tf
from keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Dropout
from matplotlib import pyplot as plt

def load_preprocess_data():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_train = (x_train - 127.5) / 127.5 #  Normalize the images to [-1, 1]

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset, x_train, y_train, x_test, y_test



