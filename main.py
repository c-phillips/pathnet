#!.venv/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

from neural_net import NeuralNetwork
from nn_trainer import NetTrainer
from pathnet import Pathnet

EPOCHS = 50
BATCH_SIZE = 32

# Load the training data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Flatten and normalize the input data because we are using linear input layers
# flatten = lambda d: d.reshape(d.shape[0], d.shape[1]*d.shape[2])
# x_train = flatten(x_train)/255
# x_test = flatten(x_test)/255
x_train = x_train/255
x_test = x_test/255

# Reformat the output data with onehot encoding
num_classes = max(y_train)+1
onehot_num = lambda y, n: np.array(list(map(lambda x: [0 if i!=x else 1 for i in range(n)], y))).reshape(y.shape[0], num_classes)
y_train = onehot_num(y_train, num_classes)
y_test = onehot_num(y_test, num_classes)

# Get the shape of our data to construct tensors
data_dims = x_train[0].shape
output_dims = y_train[0].shape
""" Example for training a single network
structure = [
	{
		'name': 'IN_Flatten',
		'layer_type': 'flatten'
	},
	{
		'name': 'Linear1',
		'num_nodes':128,
	},
	{
		'name': 'Linear2',
		'num_nodes':50,
	},
	{
		'name': 'Output',
		'num_nodes':10,
		'activation':tf.nn.softmax
	}
]

net = NeuralNetwork(structure)
trainer = NetTrainer(net, loss_function=tf.losses.softmax_cross_entropy, optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.5))# optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01))#
trainer.train(x_train, y_train, x_test, y_test, batch=BATCH_SIZE, epochs=EPOCHS)
"""
L = 3
M = int(6)
N = 3
T = 50

config = {
	"datashape": data_dims,
	"L1":{
		"num_modules":M,
		"module_structure":[
			{
				"name":"linear",
				"num_nodes":50
			}
		]
	},
	"L2":{
		"num_modules":M,
		"module_structure":[
			{
				"name":"linear",
				"num_nodes":50
			}
		]
	},
	"L3":{
		"num_modules":M,
		"module_structure":[
			{
				"name":"linear",
				"num_nodes":50
			}
		]
	}
}
# sess = tf.InteractiveSession()
PN = Pathnet(config, N)

mat = []
for i in range(L):
	indices = random.sample(range(M), N)
	row = [0]*M
	for ind in indices:
		row[ind] = 1
	mat.append(row)

path = np.array(mat)

sess = tf.InteractiveSession()
PN.train(sess, x_train, y_train, tf.losses.softmax_cross_entropy, tf.train.AdamOptimizer(learning_rate=0.05), path, T, 32)