"""Net Trainer

This module holds the class to train a neural network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys

from neural_net import NeuralNetwork


class NetTrainer:
	"""This class takes a network and trains it against a provided dataset
	using the specified loss function and optimizer.
	"""

	def __init__(self, net, loss_function=tf.losses.mean_squared_error, optimizer=tf.train.AdamOptimizer(learning_rate=0.001)):
		self.nn = net
		self.loss = loss_function
		self.optimizer = optimizer

	def train(self, x_train, y_train, x_test, y_test, batch=64, epochs=20):
		"""Trains the network for a number of epochs
		"""
		self.nn.X.set_shape([None, *x_train.shape[1:]])
		self.nn.x_input.set_shape([None, *x_train.shape[1:]])
		self.nn.Y.set_shape([None, *y_train.shape[1:]])
		self.nn.y_input.set_shape([None, *y_train.shape[1:]])
		
		self.nn.build_network()

		print("Initializing training operations...")
		loss_op = self.loss(self.nn.y_input, self.nn.yhat)
		opt_op = self.optimizer.minimize(loss_op)
		accuracy, accuracy_op = tf.metrics.accuracy(labels=self.nn.y_input, predictions=self.nn.yhat)
		
		bar_width = 40

		with tf.Session() as sess:
			writer = tf.summary.FileWriter("./logs/", sess.graph)

			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			print("Starting training...")
			for epoch_num in range(epochs):
				print(f"\nEpoch {epoch_num+1}/{epochs}:")
				# Initialize data for training
				sess.run(self.nn.data_iterator.initializer,
					feed_dict={
						self.nn.X:x_train,
						self.nn.Y:y_train,
						self.nn.batch_size:batch,
						self.nn.shuffle_size:x_train.shape[0]
					}
				)
				num_batches = int(x_train.shape[0]/batch)
				current_batch = 0
				loss = acc = 0
				while True:
					try:
						x_batch, y_batch = sess.run([self.nn.x_data, self.nn.y_data])
						sess.run(opt_op, feed_dict={self.nn.x_input:x_batch, self.nn.y_input:y_batch})

						current_batch += 1
						num_blocks = int(current_batch/num_batches*bar_width)
						bar_string = u"\r\u25D6"+u"\u25A9"*num_blocks+" "*(bar_width-num_blocks)+u"\u25D7: "

						if current_batch%10 == 0:
							l, a, _ = sess.run([loss_op, accuracy, accuracy_op], feed_dict={self.nn.x_input:x_batch, self.nn.y_input:y_batch})
							loss += l
							acc += a
							sys.stdout.write(bar_string+f"{loss/current_batch*10:.4f}, {acc/current_batch*10:.4f}")
						else:
							sys.stdout.write(bar_string)

						sys.stdout.flush()

					except tf.errors.OutOfRangeError:
						break

				# Validate the model against test data
				sess.run(self.nn.data_iterator.initializer,
					feed_dict={
						self.nn.X:x_test,
						self.nn.Y:y_test,
						self.nn.batch_size:batch,
						self.nn.shuffle_size:1#x_test.shape[0]
					}
				)
				loss = acc = 0
				current_batch = 0 
				num_batches = int(x_test.shape[0]/batch)
				while True:
					try:
						x_batch, y_batch = sess.run([self.nn.x_data, self.nn.y_data])
						l, a, _ = sess.run([loss_op, accuracy, accuracy_op], feed_dict={self.nn.x_input:x_batch, self.nn.y_input:y_batch})
						loss += l/num_batches
						acc += a/num_batches

					except tf.errors.OutOfRangeError:
						break
				print(f"\nValidation: L({loss:.6f}), A({acc*100:.4f}%)")

			writer.close()
			print("Training completed.")

			rows = 6
			columns = 5
			images = rows*columns
			num_correct = 0
			plt.figure(figsize=(60,60))
			for i in range(images):
				y = sess.run(self.nn.yhat, feed_dict={self.nn.x_input:np.expand_dims(x_test[i],axis=0), self.nn.y_input:np.expand_dims(y_test[i], axis=0)})
				# print(y[0])
				# print(y_test[i])
				# print("")

				class_names = [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               					'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
				cat_nums = np.arange(len(y[0]))

				imax = plt.subplot(rows, 2*columns, 2*i+1)
				imax.imshow(x_test[i].reshape(28,28))

				predax = plt.subplot(rows, 2*columns, 2*i+2)
				correct = False
				if np.argmax(y[0]) == np.argmax(y_test[i]):
					correct = True
					num_correct += 1
				predax.bar(cat_nums, y[0], color='g' if correct else 'r')
				predax.set_xticks(cat_nums)
				predax.set_xticklabels(class_names, rotation=60)

			print(f"Example accuracy: {num_correct/images*100}%")
			plt.subplots_adjust(hspace=0.4)
			plt.show()
