"""NN Layer

This module is to be used for creating layers for a tensorflow neural network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class NNLayer:
	"""This class handles the creation and maintenance of tensorflow
	variables and objects for a single layer of a neural network"""

	def __init__(self, net_scope, name, x, shape, layer_type="linear", activation=tf.nn.relu):
		self.num_nodes = shape[0]
		self.input_size = shape[1]

		self.name = name
		self.x_layer = x
		if len(self.x_layer.get_shape().as_list()) < 3:
			self.x_layer = tf.expand_dims(self.x_layer,axis=2)
		# self.x_layer = tf.reshape(self.x_layer, [-1,self.input_size,1])
		self.activation = activation

		if layer_type == "linear":
			with tf.name_scope(net_scope) as self.net_scope:
				with tf.name_scope(self.name) as self.scope:
					batch_size = tf.shape(self.x_layer)[0]
					self.weights = tf.Variable(tf.random_normal([self.num_nodes, self.input_size]), name=f"W_{self.name}")
					self.weights_tile = tf.tile(tf.expand_dims(self.weights, axis=0), multiples=[batch_size,1,1])
					self.biases = tf.Variable(tf.random_normal([self.num_nodes,1]), name=f"b_{self.name}")
					self.biases_tile = tf.tile(tf.expand_dims(self.biases, axis=0), multiples=[batch_size,1,1])

					self.y_layer = self.activation(tf.add(tf.matmul(self.weights_tile, self.x_layer), self.biases_tile))
		else:
			raise NotImplementedError(f"Cannot create a layer of type: {layer_type}")

		print(f"{name}: \t {self.y_layer.shape} = {self.weights_tile.shape}.{self.x_layer.shape}+{self.biases_tile.shape}\n   |\n   V")
