import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Flatten, LeakyReLU, Reshape, UpSampling2D,Conv2DTranspose
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam


class Generator(object):
    def __init__(self, latent_size=100):
        self.LATENT_SPACE_SIZE = latent_size
        self.latent_sapce = np.random.normal(0, 1, (self.LATENT_SPACE_SIZE,))
        self.Generator = self.dc_model()
        self.Generator.compile(loss = 'binary_crossentropy', optimizer=Adam(lr=1e-4, beta_1=0.2))

    def dc_model(self):
        model = Sequential([
            Dense(7 * 7 * 256, use_bias=False, input_shape=(self.LATENT_SPACE_SIZE,)),
            BatchNormalization(),
            LeakyReLU(),
            Reshape((7, 7, 256)),
            Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(1, 5, strides=2, padding='same', use_bias=False, activation='tanh')
        ])
        return model
