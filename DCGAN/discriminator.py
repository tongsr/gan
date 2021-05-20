import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Flatten, LeakyReLU, Reshape, UpSampling2D
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam


class Discriminator(object):
    def __init__(self, latent_size=100):
        self.discriminator = self.dc_model()
        self.optimizer = Adam(lr=0.0002, decay=8e-9)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def dc_model(self):
        model = Sequential([
            Conv2D(64, 5, 5, input_shape=(64, 64, 3), padding='same', activation=LeakyReLU(0.2)),
            Dropout(0.3),
            BatchNormalization(),
            Conv2D(128, 5, 5, padding='same', activation=LeakyReLU(0.2)),
            Dropout(0.3),
            BatchNormalization(),
            Flatten(),
            Dense(1,activation='sigmoid')
        ])
        return model