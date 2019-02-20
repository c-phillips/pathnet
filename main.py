#!.venv/bin/python

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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random, os, sys
# import multiprocessing as mp
from multiprocessing import Process, Lock, Pipe
from multiprocessing.connection import wait

import tensorflow as tf
import numpy as np
import copy

from neural_net import NeuralNetwork
from nn_trainer import NetTrainer
from pathnet import Pathnet

import matplotlib.pyplot as plt

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

# temp_dir = os.path.join(current_dir, "./tmp/")
# for f in os.listdir(temp_dir):
# 	fp = os.path.join(temp_dir, f)
# 	try:
# 		if os.path.isfile(fp):
# 			os.unlink(fp)
# 	except Exception as e:
# 		print(e)


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

L = 3
M = int(6)
N = 3
T = 20
B = 4
BATCH = 16
NUM_AGENTS = 8
mutate_prob = 1/(L*N)

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

def get_path(metrics):
	mat = []
	for i in range(L):
		indices = random.sample(range(M), N)
		row = [0]*M
		for ind in indices:
			row[ind] = 1
		mat.append(row)

	path = np.array(mat)
	return path

def parallel_func(config, p, c, i):
	sys.stdout = open(os.devnull, 'w')
	n = Pathnet(config, N)
	n.set_data((x_test, y_test, x_train, y_train))
	n.train(
			tf.losses.softmax_cross_entropy, 
			tf.train.GradientDescentOptimizer(learning_rate=0.001), 
			T, 
			BATCH, 
			path=p,
			conn=c
	)

def parallel_test(c,i):
	data = "init"
	for it in range(10):
		c.send(f"({i+1}): {data}")
		while not c.poll():
			continue
		data = c.recv()
	c.close()


def pool_test(args):
	i, v = args
	return (i, v**2)


def gen_paths(A, L, M, N):
	paths = []
	for a in range(A):
		path = []
		for l in range(L):
			layer = []
			possible_modules = list(range(M))
			for n in range(N):
				m = random.choice(list(range(len(possible_modules))))
				layer.append(possible_modules[m])
				del possible_modules[m]
			path.append(layer)
		paths.append(np.array(path))
	return paths


def mutate_paths(last_paths, scores, B, p, L, M, N):
	paths = last_paths
	for agent in range(len(scores)):
		new_path = agent
		agent_indices = list(range(len(scores)))

		for b in range(B):
			other_agent = random.choice(list(range(len(agent_indices))))
			if scores[agent_indices[other_agent]] > scores[new_path]:
				new_path = other_agent
			del agent_indices[other_agent]

		if new_path != agent:
			for l in range(L):
				for n in range(N):
					if np.random.rand() < p:
						paths[agent][l,n] += np.random.random_integers(-2,2)
						np.clip(paths[agent], 0, M-1, out=paths[agent])
						timeout = 0
						while np.unique(paths[agent][l,:]).shape[0] < N:
							timeout += 1
							if timeout > 50:
								print(paths[agent])
								assert False, "Module selection timeout... shouldn't"
							paths[agent][l,n] += np.random.random_integers(-2,2)
							np.clip(paths[agent], 0, M-1, out=paths[agent])

	return paths


def convert_path(path):
	converted = np.zeros((L,M))
	for r in range(path.shape[0]):
		for c in range(path.shape[1]):
			converted[r,path[r,c]] = 1
	return converted


if __name__ == '__main__':
	print("starting the processes...")
	print(f"There are {NUM_AGENTS} agents!")

	paths = gen_paths(NUM_AGENTS, L, M, N)
	send_paths = list(map(convert_path, paths))

	agents = []
	for agent in range(NUM_AGENTS):
		p, c = Pipe(duplex=True)
		proc = Process(target=parallel_func, args=(config, send_paths[agent], c, agent))

		agents.append({
			'id':agent,
			'pipe':p,
			'proc':proc
		})

		proc.start()
		c.close()
	
	i = 0
	j = 0
	num_active = NUM_AGENTS
	next_batch = False
	scores = []
	max_scores_list = []
	print("\n"+"-"*10+" Iteration {} ".format(j)+"-"*10)
	while num_active > 0:
		for agent in agents:
			p = agent['pipe']
			try:
				if p.poll():
					try:
						scores.append(p.recv())
						print(f"({agent['id']}):\t"+scores[-1])
					except EOFError:
						print("EOF")
					finally:
						i += 1
						if i >= NUM_AGENTS:
							i = 0
							j += 1
							next_batch = True
			except:
				num_active -= 1

		if next_batch:
			print("generating paths...")
			paths = mutate_paths(paths, scores, B, mutate_prob, L, M, N)
			send_paths = list(map(convert_path, paths))
			max_scores_list.append(max(scores))
			scores.clear()
			print("staring next iteration...")
			for a, agent in enumerate(agents):
				agent['pipe'].send((j, send_paths[a]))
			next_batch = False
			print("\n"+"-"*10+" Iteration {} ".format(j)+"-"*10)

	for agent in agents:
		agent['proc'].join()

	print("Done!")
	plt.plot(max_scores_list)
	plt.show()