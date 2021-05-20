import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Flatten, LeakyReLU, Reshape, UpSampling2D
from tensorflow.keras import Sequential


class Generator(object):
    def __init__(self, latent_size=100):
        self.LATENT_SPACE_SIZE = latent_size
        self.latent_sapce = np.random.normal(0, 1, (self.LATENT_SPACE_SIZE,))
        self.Generator = self.dc_model()

    def dc_model(self):
        model = Sequential([
            Dense(256*8*8, activation=LeakyReLU(0.2), input_dim=self.LATENT_SPACE_SIZE),
            BatchNormalization(),
            Reshape((8, 8, 256)),
            UpSampling2D(),
            Conv2D(128, 5, 5, padding='same', activation=LeakyReLU(0.2)),
            BatchNormalization(),
            UpSampling2D(),
            Conv2D(64, 5, 5, padding='same', activation=LeakyReLU(0.2)),
            BatchNormalization(),
            UpSampling2D(),
            Conv2D(3, 5, 5, padding='same', activation='tanh')
        ])
        return model