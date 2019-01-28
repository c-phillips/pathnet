"""Pathent

This module holds the class that holds smaller neural net modules, and controls the overall structure.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys

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

    The pathnet structure is as follows:

             L1           L2           L3
             ┌───┐        ┌───┐        ┌───┐     
           ╭─┤ █ ├───╮  ╭─┤ █ ├───╮  ╭─┤ █ ├───╮  
           |╭┤ ░ ├──╮| ╭──┤ █ ├──╮| ╭──┤ ░ ├──╮| 
    Input ─┼─┤ ░ ├───▓────┤ ░ ├───▓────┤ █ ├───▓──> Out
           |╰┤ █ ├──╯| ╰──┤ ░ ├──╯| ╰──┤ ░ ├──╯| 
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
    def __init__(self, config):
        # first, we will instantiate the networks for each pathnet layer
        self.network_structure = []
        for pn_layer in config:
            new_layer = []
            for i in range(pn_layer['num_modules']):
                new_layer.append(NeuralNetwork(pn_layer['module_structure']))
            self.network_structure.append(new_layer)

        # then, each layer is followed by a summation over all layer modules
        self.sums = []
        for i, pn_layer in enumerate(self.network_structure):
            s = pn_layer[0].yhat
            for j in range(len(pn_layer)-1):
                s += pn_layer[j+1].yhat
            self.sums.append(s)
            # if this is not the first layer, we will set the input of each
            # module to the last pn_layer summation
            if i > 0:
                for module in pn_layer:
                    module.x_input = self.sums[i-1]

        self.output = self.sums[-1] # the main network output is the last sum layer
        