"""Net Trainer

This module holds the class to train a neural network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sys

from neural_net import NeuralNetwork


class NetTrainer:
	"""This class takes a network and trains it against a provided dataset
	using the specified loss function and optimizer.
	"""

	def __init__(self, net, loss_function=tf.losses.mean_squared_error, optimizer=tf.train.AdamOptimizer(learning_rate=0.1)):
		self.nn = net
		self.loss = loss_function
		self.optimizer = optimizer

	def train(self, x_train, y_train, batch=64, epochs=20):
		"""Trains the network for a number of epochs
		"""

		loss_op = self.loss(self.nn.y_data, self.nn.yhat)
		opt_op = self.optimizer.minimize(loss_op)
		accuracy_op = tf.metrics.accuracy(self.nn.y_data, self.nn.yhat)

		with tf.Session() as sess:
			writer = tf.summary.FileWriter("./logs/", sess.graph)

			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())

			for i in range(epochs):
				sess.run(self.nn.data_iterator.initializer, feed_dict={self.nn.X:x_train, self.nn.Y:y_train, self.nn.batch_size:batch})
				avg_loss = avg_acc = 0
				num_batches = int(x_train.shape[0]/batch)
				current_batch = 0
				while True:
					try:
						sess.run([self.nn.x_data, self.nn.y_data])
						_, loss, accuracy = sess.run([opt_op, loss_op, accuracy_op])
						avg_loss += loss/num_batches
						avg_acc += accuracy[0]/num_batches
						current_batch += 1
						sys.stdout.write(f"\r{current_batch}/{num_batches}:\t")
						sys.stdout.flush()
					except tf.errors.OutOfRangeError:
						break

				epoch_divisor = int(epochs/10)
				if (i%epoch_divisor) == 0:
					print(f"E({i}): {avg_loss}, {avg_acc}\n\n")

			writer.close()
