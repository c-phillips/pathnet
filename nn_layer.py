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
		self.input_size = shape[1:]
		self.type = layer_type

		self.name = name
		self.x_layer = x
		self.activation = activation

		# with tf.variable_scope(net_scope) as self.net_scope:
		with tf.variable_scope(self.name) as self.scope:
			if layer_type == "linear":
				self.weights = self.get_var(f"W", initializer=tf.truncated_normal([*self.input_size[1:], self.num_nodes], mean=0.0, stddev=0.02)) 
				self.biases = self.get_var(f"b", initializer=tf.truncated_normal([self.num_nodes], mean=0.0, stddev=0.02))

				self.y_layer = self.activation(tf.add(tf.matmul(self.x_layer, self.weights), self.biases))
				self.attach_summaries(self.y_layer)

			elif layer_type == "flatten":
				self.y_layer = tf.layers.flatten(self.x_layer)

			elif layer_type == "conv2d":
				self.y_layer = tf.nn.conv()

			else:
				raise NotImplementedError(f"Cannot create a layer of type: {layer_type}")

	def get_var(self, *args, **kwargs):
		# tf.get_variable(f"W_{self.name}", initializer=tf.truncated_normal([*self.input_size[1:], self.num_nodes], mean=0.0, stddev=0.02))
		v = tf.get_variable(*args, **kwargs)
		self.attach_summaries(v, args[0])
		return v

	def attach_summaries(self, var, scope="summaries"):
		"""Attaches summaries for tensorboard visualization.
		"""
		with tf.name_scope(scope):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)
