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
		if type(self.input_size) is not list:
			weights_dims = [self.input_size, self.num_nodes]
		else:
			weights_dims = [*self.input_size, self.num_nodes]

		self.name = name
		self.x_layer = x
		self.activation = activation

		if layer_type == "linear":
			with tf.name_scope(net_scope) as self.net_scope:
				with tf.name_scope(self.name) as self.scope:
					# self.weights = tf.Variable(tf.random_normal(weights_dims), name=f"W_{self.name}")
					# self.biases = tf.Variable(tf.random_normal([self.num_nodes]), name=f"b_{self.name}")
					self.weights = tf.get_variable(f"W_{self.name}", initializer=tf.truncated_normal(weights_dims, mean=0.0, stddev=0.02))
					self.biases = tf.get_variable(f"b_{self.name}", initializer=tf.truncated_normal([self.num_nodes], mean=0.0, stddev=0.02))

					self.y_layer = self.activation(tf.add(tf.matmul(self.x_layer, self.weights), self.biases))
		else:
			raise NotImplementedError(f"Cannot create a layer of type: {layer_type}")

		print(f"{name}: \t {self.y_layer.shape} = {self.x_layer.shape}.{self.weights.shape}+{self.biases.shape}\n   |\n   V")
