"""Pathent

This module holds the class that holds smaller neural net modules, and controls the overall structure.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys, copy

from neural_net import NeuralNetwork
from nn_layer import NNLayer

class Pathnet:
    """This class contains the main structure for pathnet. At initialization time,
    a dictionary should be passed that describes the configuration of each overall
    network layer, including what modules are in each.

    The config dicationary has the form:
    ```python
    config = {
        'LayerName':{
            'num_modules':6,
            'module_structure':[
                {
                    'name':'Linear1',
                    'layer_type':'linear',
                    'activation':tf.nn.sigmoid, # Optional (default: tf.nn.relu)
                    'ouput_size':10
                }
            ]
        }
    }
    ```
    The class should also be initialized with the number of active modules (N) 
    allowed for each layer. 

    The pathnet structure is as follows:

             L1           L2           L3
             ┌───┐        ┌───┐        ┌───┐     
           ╭─┤ █ ├───╮  ╭─┤ █ ├───╮  ╭─┤ █ ├───╮  
           |╭┤ ░ ├──╮| ╭┴─┤ █ ├──╮| ╭┴─┤ ░ ├──╮| 
    Input ─┼┼┤ ░ ├───▓─┼──┤ ░ ├───▓─┼──┤ █ ├───▓──> Out
           |╰┤ █ ├──╯| ╰┬─┤ ░ ├──╯| ╰┬─┤ ░ ├──╯| 
           ╰─┤ █ ├───╯  ╰─┤ █ ├───╯  ╰─┤ █ ├───╯  
             └───┘        └───┘        └───┘     
    █ active module
    ░ inactive module
    ▓ module summation

    The path is the activation of modules. Inactive modeles do not contribute to 
    the gradient calculations for optimization. 

    An approximation of the optimal path is learned by genetic algorithm for the 
    moment.
    """
    def __init__(self, config, N):
        # first, we will instantiate the networks for each pathnet layer
        self.network_structure = []

        data_dims = list(config.pop("datashape", None))
        if data_dims is None:
            raise ValueError("You must provide a datashape parameter to configure PathNet!")
        data_dims.insert(0, None)
        print(data_dims)

        for layer_name, layer_structure in config.items():
            new_layer = []
            for i in range(layer_structure['num_modules']):
                # if it is the first module of the layer, we set the x_input to None for
                # assignment later; otherwise, we set it to the input of the first module
                temp_struct = copy.deepcopy(layer_structure['module_structure'])
                for j in range(len(temp_struct)):
                    if 'name' in temp_struct[j]:
                        temp_struct[j]['name'] += "_{}_{}".format(layer_name, i+1)
                        print(temp_struct[j]['name'])
                new_layer.append(NeuralNetwork(temp_struct,
                                               x_in = None if i == 0 else new_layer[0].x_input,
                                               x_in_shape = data_dims if i == 0 else None,
                                               first_input = True if len(self.network_structure) == 0 else False))
                del temp_struct
                tf.get_variable_scope().reuse_variables()
            self.network_structure.append(new_layer)

        self.L = len(self.network_structure)
        self.M = len(self.network_structure[0])
        self.N = N

        self.Pmat = tf.placeholder(tf.float32, [self.L, self.M], name="PathMatrix")

        # this just makes it simpler to reference the first module since it is really the
        # arbiter of the training data. It holds the primary dataset information and feeds
        # not only the network, but the batched data to the optimizer
        self.network_structure[0][0].build_network()
        self.fm = self.network_structure[0][0]

        # then, each layer is followed by a summation over all layer modules
        with tf.variable_scope("PathNet", reuse=tf.AUTO_REUSE) as self.var_scope:
            self.sums = []
            for i, pn_layer in enumerate(self.network_structure):
                self.sums.append(tf.get_variable("sum_layer_{}".format(i),
                                shape=self.sums[-1].get_shape().as_list() if i > 0 else self.fm.yhat.get_shape().as_list(),    # hopefully tensorflow makes this work
                                initializer=tf.zeros_initializer))
                for j in range(len(pn_layer)):
                    pn_layer[j].build_network()
                    # if this is the first module, not in the first layer, we assign
                    # the x_input as the output of the last summation module
                    if j == 0 and i > 0:
                        pn_layer[j].x_input = self.sums[-1]
                    self.sums[-1] = self.sums[-1]+self.Pmat[i,j]*pn_layer[j].yhat
                # self.sums.append(s)

            self.output = self.sums[-1] # the main network output is the last sum layer
    
    def train(self, sess, x_train, y_train, loss_func, opt_func, path, T, batch):
        """This method is used to train the individual pathnet agent.
        A session reference must be passed, as well as the path to train over.

        Args:
            sess:       the tensorflow session to run operations
            x_train:    the input training data
            y_train:    the output training data
            loss_func:  the tensorflow loss function to use
            opt_func:   the tensorflow optimization function to use
            path:       the module selection path for directed training
            T:          the number of epochs over which to optimize
            batch:      the batch size to train with

        The path must be an LxM matrix correspoding to the modules to train.
        The function will then feed the appropriate values through to the Pmat
        placeholder, and select the proper variables to optimize over
        """
        with tf.variable_scope(self.var_scope, reuse=tf.AUTO_REUSE):
            # we need to store the variables we need to optimize over according to the path
            train_vars = []
            for l in range(path.shape[0]):
                for m in range(path.shape[1]):
                    if path[l,m] != 0:
                        for layer in self.network_structure[l][m].layers:
                            # naturally this will need to change if the structure of the layer
                            # does not use weights and biases
                            train_vars.append(layer.weights)
                            train_vars.append(layer.biases)
            
            loss_op = loss_func(self.fm.y_input, self.output)
            opt_op = opt_func.minimize(loss_op, var_list=train_vars)
            accuracy, accuracy_op = tf.metrics.accuracy(labels=self.fm.y_input, predictions=self.output)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for epoch_num in range(T):
                print(f"\nEpoch {epoch_num+1}/{T}:")
                # Initialize data for training
                sess.run(self.fm.data_iterator.initializer,
                    feed_dict={
                        self.fm.X:x_train,
                        self.fm.Y:y_train,
                        self.fm.batch_size:batch,
                        self.fm.shuffle_size:x_train.shape[0]
                    }
                )
                num_batches = int(x_train.shape[0]/batch)
                # current_batch = 0
                loss = acc = 0
                while True:
                    try:
                        x_batch, y_batch = sess.run([self.fm.x_data, self.fm.y_data])
                        sess.run(opt_op, feed_dict={self.fm.x_input:x_batch, self.fm.y_input:y_batch, self.Pmat:path})

                        # current_batch += 1
                        # num_blocks = int(current_batch/num_batches*bar_width)
                        # bar_string = u"\r\u25D6"+u"\u25A9"*num_blocks+" "*(bar_width-num_blocks)+u"\u25D7: "

                        # if current_batch%10 == 0:
                        #     l, a, _ = sess.run([loss_op, accuracy, accuracy_op], feed_dict={self.fm.x_input:x_batch, self.fm.y_input:y_batch})
                        #     loss += l
                        #     acc += a
                        #     sys.stdout.write(bar_string+f"{loss/current_batch*10:.4f}, {acc/current_batch*10:.4f}")
                        # else:
                            # sys.stdout.write(bar_string)

                        # sys.stdout.flush()

                    except tf.errors.OutOfRangeError:
                        break