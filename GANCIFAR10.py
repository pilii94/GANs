import logging

import os


from tensorflow.keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

import tensorflow as tf
from tensorflow import keras
from numpy.random import randint

from numpy import ones
from numpy import zeros

from numpy.random import randn
from numpy.random import randint

from matplotlib import pyplot
from config import config
import colored

os.environ["CUDA_VISIBLE_DEVICES"]="4,5"


class SimpleGAN():
	def __init__(self, logger=None):
		self.logger = logger
		self.cifar = keras.datasets.cifar10
		self.n_nodes = 256 * 4 * 4
		

	def create_discriminator(self, in_shape=(32,32,3)):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=in_shape))
		model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
		model.add(tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
		model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
		model.add(tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
		model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
		model.add(tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same'))
		model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dropout(0.4))
		model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

		# compile discriminator
		opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
		model.compile(loss=config['loss_discr'], optimizer=opt, metrics=['accuracy'])
		return model

	def create_generator(self, latent_dim):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Dense(self.n_nodes, input_dim=latent_dim))
		model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
		model.add(tf.keras.layers.Reshape((4, 4, 256)))
		model.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
		model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
		model.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
		model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
		model.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
		model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
		
		model.add(tf.keras.layers.Conv2D(3, (3,3), activation='tanh', padding='same'))
		return model


	def create_gan(self, g_model, d_model):
		'''combine generator and discriminator'''
		# train discriminator and generator separately
		d_model.trainable = False 
		model = tf.keras.Sequential()
		# add generator
		model.add(g_model)
		# add discriminator
		model.add(d_model)
		# compile 
		opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
		model.compile(loss=config['loss_gan'], optimizer=opt)
		return model


	def load_real_samples(self):
		(trainX, _), (_, _) = self.cifar.load_data()
		X = trainX.astype('float32')
		# scale
		X = (X - 127.5) / 127.5
		return X

	def generate_real_samples(self, dataset, n_samples):
		''' 
		take sample from CIFAR10 dataset 
		'''
		ix = randint(0, dataset.shape[0], n_samples)
		X = dataset[ix]
		y = ones((n_samples, 1))
		return X, y


	def generate_latent_points(self, latent_dim, n_samples):
		''' generator input '''
		x_input = randn(latent_dim * n_samples)
		x_input = x_input.reshape(n_samples, latent_dim)
		return x_input


	def generate_fake_samples_with_generator(self, g_model, latent_dim, n_samples):
		x_input = self.generate_latent_points(latent_dim, n_samples)
		X = g_model.predict(x_input)
		y = zeros((n_samples, 1))
		return X, y


	def generate_plot(self, examples, epoch, n=7):
		# plot generated images
		examples = (examples + 1) / 2.0
		
		for i in range(n * n):
			pyplot.subplot(n, n, 1 + i)
			pyplot.axis('off')
			pyplot.imshow(examples[i])
		filename = f'generated_plot_e{epoch+1}.jpg'
		pyplot.savefig(filename)
		pyplot.close()


	def evaluate_gan_performance(self, epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
		'''
		Evaluate generator and discriminator and plot images generated
		'''
		X_real, y_real = self.generate_real_samples(dataset, n_samples)
		_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
		x_fake, y_fake = self.generate_fake_samples_with_generator(g_model, latent_dim, n_samples)
		_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
		self.logger.info(f'>Accuracy real: {acc_real*100}, fake: {acc_fake*100}')
		self.generate_plot(x_fake, epoch)
		filename = f'generator_model_{epoch+1}.h5'
		g_model.save(filename)


	def train_gan(self, g_model, d_model, gan_model, dataset, latent_dim, n_epochs=50, n_batch=128):
		'''
		train the generator and discriminator
		'''
		bat_per_epo = int(dataset.shape[0] / n_batch)
		half_batch = int(n_batch / 2)
		# manually enumerate epochs
		for i in range(n_epochs):
			for j in range(bat_per_epo):
				X_real, y_real = self.generate_real_samples(dataset, half_batch)
				d_loss1, _ = d_model.train_on_batch(X_real, y_real)
				X_fake, y_fake = self.generate_fake_samples_with_generator(g_model, latent_dim, half_batch)
				d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
				X_gan = self.generate_latent_points(latent_dim, n_batch)
				y_gan = ones((n_batch, 1))
				g_loss = gan_model.train_on_batch(X_gan, y_gan)
				self.logger.info(f'>{i+1}, {j+1}/{bat_per_epo}, d1={d_loss1}, d2={d_loss2}, g={g_loss}')
			if (i+1) % 10 == 0:
				self.evaluate_gan_performance(i, g_model, d_model, dataset, latent_dim)


if __name__ == '__main__':

	# Logging
	logger = logging.getLogger('GAN_CIFAR10')
	logger.setLevel(logging.DEBUG)

	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.DEBUG)

	logger.addHandler(console_handler)

	gan = SimpleGAN(logger=logger)

	d_model = gan.create_discriminator()
	g_model = gan.create_generator(config['latent_dim'])
	gan_model = gan.create_gan(g_model, d_model)
	dataset = gan.load_real_samples()

	logger.info("Training GAN...")
	gan.train_gan(g_model, d_model, gan_model, dataset, config['latent_dim'], n_epochs=config['epochs'])

	logger.info(f'Done model training PID: ' +
				colored(f'{os.getpid()}', 'green'))

	






