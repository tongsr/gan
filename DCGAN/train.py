# https://github.com/marload/GANs-TensorFlow2/blob/master/DCGAN/DCGAN.py


import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Flatten, LeakyReLU, Reshape, UpSampling2D
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
import discriminator
import generator
import random
import copy

latent_size = 100
latent_space = np.random.normal(0, 1, (latent_size,))
generator = generator.Generator(latent_size=latent_size).Generator
discriminator = discriminator.Discriminator(latent_size=latent_size).discriminator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = np.array(x_train,dtype=float)
x_train = np.expand_dims(x_train, -1)
x_train = (x_train - 127.5) / 127.5

Batch_Size = 100

generator.summary()
#discriminator.summary()

# generator = tf.keras.models.load_model('dcgan.h5')
# discriminator = tf.keras.models.load_model('discriminator.h5')
discriminator.trainable = False

gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy',optimizer=Adam(lr=2e-3, decay=8e-9))


def flipcoin(chance=0.5):
    return np.random.binomial(1, chance)


def sample_latent_space(instances):
    return np.random.normal(0, 1, (instances, latent_size))


def train():
    count = 0
    x_train_temp = copy.deepcopy(x_train)
    while len(x_train_temp) > Batch_Size:
        count = count + 1
        if flipcoin():
            count_real_images = Batch_Size
            start_index = random.randint(0, (len(x_train_temp) - count_real_images))
            real_image_raw = x_train_temp[start_index:(start_index + count_real_images)]
            x_train_temp = np.delete(x_train_temp, range(start_index, (start_index + count_real_images)), 0)
            x_batch = np.array(real_image_raw)
            y_batch = np.ones([count_real_images, 1])
        else:
            latent_space_samples = sample_latent_space(Batch_Size)
            x_batch = generator.predict(latent_space_samples)
            y_batch = np.zeros([Batch_Size, 1])

        discriminator_loss = discriminator.train_on_batch(x_batch, y_batch)

        if flipcoin(0.9):
            y_gen_label = np.ones([Batch_Size, 1])
        else:
            y_gen_label = np.zeros([Batch_Size, 1])

        x_latent_space_sample = sample_latent_space(Batch_Size)
        gan_loss = gan.train_on_batch(x_latent_space_sample, y_gen_label)

    print('epoch:' + str(count) + '  loss1:' + str(discriminator_loss) + '  gan_loss:' + str(gan_loss))
    generator.save('dcgan.h5')
    discriminator.save('discriminator.h5')


for i in range(20):
    train()

