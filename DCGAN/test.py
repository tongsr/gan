import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Flatten, LeakyReLU, Reshape, UpSampling2D
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist

import random
import copy
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('dcgan.h5')


def sample_latent_space(instances):
    return np.random.normal(0, 1, (instances, 100))


s = sample_latent_space(16)

images = model.predict(s)
file_name = 'result.png'

plt.figure(figsize=(10, 10))
for i in range(images.shape[0]):
    plt.subplot(4, 4, i+1)
    image = images[i, :, :, :]
    image = np.reshape(image,[28,28])
    plt.imshow(image)
    plt.axis('off')
plt.tight_layout()
plt.savefig(file_name)
plt.close('all')