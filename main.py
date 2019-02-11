#!.venv/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random, os

import numpy as np
import tensorflow as tf

from neural_net import NeuralNetwork
from nn_trainer import NetTrainer
from pathnet import Pathnet

# remove logs from past runs
current_dir = os.path.dirname(os.path.realpath(__file__))

train_dir = os.path.join(current_dir, "./logs/train")
for f in os.listdir(train_dir):
	fp = os.path.join(train_dir, f)
	try:
		if os.path.isfile(fp):
			os.unlink(fp)
	except Exception as e:
		print(e)

test_dir = os.path.join(current_dir, "./logs/test")
for f in os.listdir(test_dir):
	fp = os.path.join(test_dir, f)
	try:
		if os.path.isfile(fp):
			os.unlink(fp)
	except Exception as e:
		print(e)


# Load the training data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Flatten and normalize the input data because we are using linear input layers
# flatten = lambda d: d.reshape(d.shape[0], d.shape[1]*d.shape[2], 1)
# x_train = x_train.flatten()/255
# x_test = x_test.flatten()/255
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
T = 10
BATCH = 16

config = {
	"datashape": data_dims,
	"FL":{
		"num_modules":1,
		"conditioning": True,
		"module_structure":[
			{
				"name":"flatten",
				"layer_type":"flatten"
			}
		]
	},
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
				"num_nodes":10
			}
		]
	}
}
# sess = tf.InteractiveSession()
PN = Pathnet(config, N)

def get_path(accuracy):
	mat = []
	for i in range(L):
		indices = random.sample(range(M), N)
		row = [0]*M
		for ind in indices:
			row[ind] = 1
		mat.append(row)

	path = np.array(mat)
	print(path)

PN.train(x_train, y_train, x_test, y_test, tf.losses.softmax_cross_entropy, tf.train.GradientDescentOptimizer(learning_rate=0.001), T, BATCH, path_func=get_path)