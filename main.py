#!.venv/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from neural_net import NeuralNetwork
from nn_trainer import NetTrainer

EPOCHS = 20
BATCH_SIZE = 32

# Load the training data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Flatten and normalize the input data because we are using linear input layers
flatten = lambda d: d.reshape(d.shape[0], d.shape[1]*d.shape[2])
x_train = flatten(x_train)/255
x_test = flatten(x_test)/255

# Reformat the output data with onehot encoding
num_classes = max(y_train)+1
onehot_num = lambda y, n: np.array(list(map(lambda x: [0 if i!=x else 1 for i in range(n)], y))).reshape(y.shape[0], num_classes,1)
y_train = onehot_num(y_train, num_classes)
y_test = onehot_num(y_test, num_classes)

# Get the shape of our data to construct tensors
data_dims = x_train[0].shape
output_dims = y_train[0].shape

structure = [
	{
		'name': 'Linear1',
		'input_size':data_dims[0],
		'num_nodes':10,
	},
	{
		'name': 'Linear2',
		'num_nodes':15,
	},
	{
		'name': 'Output',
		'num_nodes':10,
		'activation':tf.nn.softmax
	}
]

net = NeuralNetwork(structure)
trainer = NetTrainer(net)
trainer.train(x_train, y_train, batch=BATCH_SIZE, epochs=EPOCHS)