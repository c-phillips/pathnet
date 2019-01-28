"""Neural Net

This module holds the class to build a neural network with layers
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nn_layer import NNLayer


class NeuralNetwork:
	"""This class provides the interfaces to add layers and train
	the weights given an input
	"""

	def __init__(self, network_structure=None):
		print("Initializing Network...")
		self.layers = []
		self.structure = network_structure

		self.output_size = network_structure[-1]['num_nodes']

		self.X = tf.placeholder(tf.float32, name="Input_Data")
		self.Y = tf.placeholder(tf.float32, name="Data_Labels")

		self.x_input = tf.placeholder(tf.float32, name="network_input")
		self.y_input = tf.placeholder(tf.float32, name="network_output")

		with tf.name_scope("Dataset"):
			self.batch_size = tf.placeholder(tf.int64)
			self.shuffle_size = tf.placeholder(tf.int64)
			self.epochs = tf.placeholder(tf.int64)

			print("Creating dataset pipeline...")
			self.dataset = tf.data.Dataset.from_tensor_slices((self.X,self.Y))
			self.dataset = self.dataset.shuffle(self.shuffle_size)
			self.dataset = self.dataset.batch(self.batch_size)
			# self.dataset = self.dataset.repeat(self.epochs)
			# self.dataset = self.dataset.prefetch(1)
			self.data_iterator = self.dataset.make_initializable_iterator()
			self.x_data, self.y_data  = self.data_iterator.get_next()

	def build_network(self):
		print("Building Network structure...")
		if self.structure is not None:
			with tf.name_scope("Network") as self.name_scope:
				for layer in self.structure:
					self.add_layer(**layer)
		else:
			print("Must provide the NeuralNetwork with a structure")
			exit()

	def add_layer(self, name=None, num_nodes=1, input_size=0, layer_type="linear", activation=tf.nn.relu):
		"""This function creates and stores layers
		"""
		if len(self.layers) > 0:
			input_size = self.layers[-1].y_layer.get_shape().as_list()
			x_layer = self.layers[-1].y_layer
		else:
			input_size = self.x_input.get_shape().as_list()
			x_layer = self.x_input

		if name is not None:
			layer_name = name
		else:
			layer_name = layer_type+str(len(self.layers))

								 #    net_scope,       name,       x,             shape,           layer_type="linear",   activation=tf.nn.relu)
		self.layers.append(NNLayer(self.name_scope, layer_name, x_layer, [num_nodes, *input_size], layer_type=layer_type, activation=activation))
		self.yhat = self.layers[-1].y_layer
		
		return self.layers[-1]
