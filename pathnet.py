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
            
bar_width = 25

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
        # with tf.variable_scope("PathNet", reuse=tf.AUTO_REUSE) as self.var_scope:
        # first, we will instantiate the networks for each pathnet layer
        self.network_structure = []
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        data_dims = list(config.pop("datashape", None))
        if data_dims is None:
            raise ValueError("You must provide a datashape parameter to configure PathNet!")
        data_dims.insert(0, None)
        print(data_dims)

        self.M = 0
        self.L = 0
        self.N = N

        self.first_layer = None
        self.sums = []
        for l, (layer_name, layer_structure) in enumerate(config.items()):
            new_layer = []
            
            if 'conditioning' not in layer_structure:
                print(layer_name)
                if self.M == 0:
                    self.M = layer_structure['num_modules']
                    self.first_layer = l
                    self.L = len(config.items())-l
                    print("FIRST LAYER: {}".format(layer_name))

                    self.Pmat = tf.placeholder(tf.float32, [self.L, self.M], name="PathMatrix")
                    print(self.Pmat.get_shape().as_list())
            else:
                print("CONDITIONING: {}".format(layer_name))


            with tf.variable_scope(layer_name):
                for i in range(layer_structure['num_modules']):
                    # if it is the first module of the layer, we set the x_input to None for
                    # assignment later; otherwise, we set it to the input of the first module
                    temp_struct = copy.deepcopy(layer_structure['module_structure'])
                    # for j in range(len(temp_struct)):
                    #     if 'name' in temp_struct[j]:
                    #         temp_struct[j]['name'] += "_{}_{}".format(layer_name, i+1)
                    #         # print(temp_struct[j]['name'])

                    # We need to select the proper input for this network provided the other
                    # networks/layers have been created. 
                    input_ref = None
                    if i == 0 and self.first_layer is not None and l > self.first_layer:
                        input_ref = self.sums[-1]
                    elif i == 0 and self.first_layer is not None and l == self.first_layer:
                        input_ref = self.network_structure[self.first_layer-1][0].yhat
                    elif i == 0 and self.first_layer is None and l > 0:
                        input_ref = self.network_structure[l-1].yhat
                    elif i > 0 and self.first_layer is not None:
                        input_ref = new_layer[0].x_input
                        
                    with tf.variable_scope("M"+str(i)):
                        new_layer.append(
                            NeuralNetwork(temp_struct,
                                        x_in = input_ref,
                                        x_in_shape = data_dims if l == 0 else self.network_structure[l-1][0].yhat.get_shape().as_list(),
                                        make_dataset = True if l == 0 else False,
                                        name = "M"+str(i)
                            )
                        )
                        new_layer[-1].build_network()

                    del temp_struct
                    if 'conditioning' not in layer_structure and i == 0:
                        sum_shape = new_layer[-1].yhat.get_shape().as_list() # self.sums[-1].get_shape().as_list() if l > self.first_layer else 
                        # with tf.variable_scope("sums_{}".format(layer_name), reuse=tf.AUTO_REUSE):
                        s = tf.get_variable("sum", [*sum_shape[1:]], initializer=tf.zeros_initializer())#, validate_shape=False)
                            # s = tf.get_variable("sum", initializer=tf.truncated_normal([*sum_shape[1:]], mean=0.0, stddev=0.0))
                        self.sums.append(s)

                    if self.first_layer is not None:
                        self.sums[-1] = self.sums[-1]+self.Pmat[l-self.first_layer,i]*new_layer[-1].yhat

                    # tf.get_variable_scope().reuse_variables()
            self.network_structure.append(new_layer)

        self.output = self.sums[-1] # the main network output is the last sum layer
        self.data_layer = self.network_structure[0][0] # this makes it easy to access our datapipeline
    
    def set_data(self, all_data):
        print(len(all_data))
        assert len(all_data) == 4, "You must provide a test set and train set with labels!"
        self.x_train = all_data[0]
        self.y_train = all_data[1]
        self.x_test = all_data[2]
        self.y_test = all_data[3]

    def train(self, loss_func, opt_func, T, batch, num_batches=None, path=None, conn=None):
        """This method is used to train the individual pathnet agent.
        A session reference must be passed, as well as the path to train over.

        Args:
            sess:       the tensorflow session to run operations
            # x_train:    the input training data
            # y_train:    the output training data
            # x_test:     the input testing data
            # y_test:     the output testing data
            loss_func:  the tensorflow loss function to use
            opt_func:   the tensorflow optimization function to use
            T:          the number of epochs over which to optimize
            batch:      the batch size to train with
            num_batches:the number of batches to train for each epoch
            path:       the path to train over
            conn:       the multiprocessing pipe

        The path must be an LxM matrix correspoding to the modules to train.
        The function will then feed the appropriate values through to the Pmat
        placeholder, and select the proper variables to optimize over
        """
        if self.x_train is None:
            print("ERR: Must call set_data before training!")
            exit()

        # get_path = path_func if path_func is not None else self.get_path
        with tf.Session() as sess:
            # we need to merge all of the variable summaries for reporting
            merged_summaries = tf.summary.merge_all()

            # here is where we initialize the summary writers
            train_writer = tf.summary.FileWriter("./logs/train", sess.graph)
            test_writer = tf.summary.FileWriter("./logs/test")

            # we have to specify placeholders for some performance metrics that we calculate ourselves
            # then we pass the scalar values through a feed_dict when we process the summaries.
            loss_ph = tf.placeholder(tf.float32, shape=None, name="loss_summary")
            loss_summary = tf.summary.scalar("loss", loss_ph)

            acc_ph = tf.placeholder(tf.float32, shape=None, name="acc_summary")
            acc_summary = tf.summary.scalar("accuracy", acc_ph)

            performance_summaries = tf.summary.merge([loss_summary, acc_summary])   # merge the custom summaries
            
            loss_op = loss_func(self.data_layer.y_input, self.output)
            
            # Here we initialize the training variable list to hold everything for the time being.
            # During processing, this list is updated to reflect the current path. The reference should be stored
            # in the opt_op so when it changes, it should properly compute the right variables. 
            train_vars = []
            for l in range(self.L):
                    for m in range(self.M):
                        for layer in self.network_structure[l+self.first_layer][m].layers:
                            train_vars.append(layer.weights)
                            train_vars.append(layer.biases)
            opt_op = opt_func.minimize(loss_op, var_list=train_vars)

            # This calculates the accuracy using the MAP from the output
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(self.data_layer.y_input, 1), tf.argmax(self.output, 1))
                with tf.name_scope('accuracy'):
                    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # accuracy, accuracy_op = tf.metrics.accuracy(labels=self.data_layer.y_input, predictions=self.output)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # we need a way to average and track performance variables accross batches
            avg = lambda x: sum(x)/len(x) if len(x) > 0 else 0
            loss = []
            acc = []
            recent_metrics = []
            metric_dict = {}    # we will add metrics to this dictionary and pass it to the path generator
            for epoch_num in range(T):
                loss.clear()
                acc.clear()
                metric_dict.clear()

                # we need to store the variables we need to optimize over according to the path
                # path = get_path(metric_dict)
                train_vars.clear()
                for l in range(path.shape[0]):
                    for m in range(path.shape[1]):
                        if path[l,m] != 0:
                            for layer in self.network_structure[l+self.first_layer][m].layers:
                                # naturally this will need to change if the structure of the layer
                                # does not use weights and biases
                                train_vars.append(layer.weights)
                                train_vars.append(layer.biases)

                print(f"\nEpoch {epoch_num+1}/{T}:")

                # Initialize data for training
                sess.run(self.data_layer.data_iterator.initializer,
                    feed_dict={
                        self.data_layer.X:self.x_train,
                        self.data_layer.Y:self.y_train,
                        self.data_layer.batch_size:batch,
                        self.data_layer.shuffle_size:self.x_train.shape[0]
                    }
                )
                num_batches = int(self.x_train.shape[0]/batch)
                current_batch = 0
                while current_batch < num_batches or num_batches is None:
                    try:
                        x_batch, y_batch = sess.run([self.data_layer.x_data, self.data_layer.y_data])
                        sess.run(opt_op, feed_dict={self.data_layer.x_input:x_batch, self.data_layer.y_input:y_batch, self.Pmat:path})
                        l, a = sess.run([loss_op, accuracy_op], feed_dict={self.data_layer.x_input:x_batch, self.data_layer.y_input:y_batch, self.Pmat:path})
                        loss.append(l)
                        acc.append(a)

                        current_batch += 1
                        num_blocks = int(current_batch/num_batches*bar_width)
                        bar_string = ""#u"\r\u25D6"+u"\u25A9"*num_blocks+" "*(bar_width-num_blocks)+u"\u25D7 "

                        if current_batch%10 == 0 and current_batch > 0:
                            performance_summary= sess.run(performance_summaries, feed_dict={loss_ph:avg(loss), acc_ph:avg(acc)})
                            train_writer.add_summary(performance_summary, current_batch+(num_batches*epoch_num))

                            var_summary = sess.run(merged_summaries, feed_dict={self.data_layer.x_input:x_batch, self.data_layer.y_input:y_batch, self.Pmat:path})
                            train_writer.add_summary(var_summary, current_batch+(num_batches*epoch_num))
                            
                            sys.stdout.write(bar_string+f": {avg(loss):.4f}, {avg(acc)*100:.2f}%")
                            
                            recent_metrics.clear()
                            recent_metrics.append(avg(acc))

                            loss.clear()
                            acc.clear()
                        else:
                            sys.stdout.write(bar_string)

                        sys.stdout.flush()

                    except tf.errors.OutOfRangeError:
                        break

                if conn is not None:
                    conn.send(str(*recent_metrics))
                    while not conn.poll():
                        continue
                iteration_number, path = conn.recv()
                assert type(path) == np.ndarray, "PATH MUST BE NUMPY ARRAY"

                # Validate the model against test data
                sess.run(self.data_layer.data_iterator.initializer,
                    feed_dict={
                        self.data_layer.X:self.x_test,
                        self.data_layer.Y:self.y_test,
                        self.data_layer.batch_size:batch,
                        self.data_layer.shuffle_size:1#x_train.shape[0]
                    }
                )
                loss.clear()
                acc.clear()
                current_batch = 0 
                num_batches = int(self.x_test.shape[0]/batch)
                while True:
                    try:
                        x_batch, y_batch = sess.run([self.data_layer.x_data, self.data_layer.y_data])
                        l, a = sess.run([loss_op, accuracy_op], feed_dict={self.data_layer.x_input:x_batch, self.data_layer.y_input:y_batch, self.Pmat:path})
                        loss.append(l)
                        acc.append(a)

                        if current_batch%10 == 0:
                            summary = sess.run(performance_summaries, feed_dict={loss_ph:avg(loss), acc_ph:avg(acc)})
                            test_writer.add_summary(summary, current_batch+(num_batches*epoch_num))
                            
                            loss.clear()
                            acc.clear()

                        current_batch += 1

                    except tf.errors.OutOfRangeError:
                        break

                print(f"\nValidation: L({avg(loss):.6f}), A({avg(acc)*100:.4f}%)")
            print("\nFinished!")
                
            test_writer.close()
            train_writer.close()

            # rows = 6
            # columns = 5
            # images = rows*columns
            # num_correct = 0
            # plt.figure()
            # for i in range(images):
            #     y = sess.run(self.output, feed_dict={self.data_layer.x_input:np.expand_dims(x_test[i],axis=0), self.data_layer.y_input:np.expand_dims(y_test[i], axis=0), self.Pmat:path})
            #     # print(y[0])
            #     # print(y_test[i])
            #     # print("")

            #     class_names = [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
            #                     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            #     cat_nums = np.arange(len(y[0]))

            #     imax = plt.subplot(rows, 2*columns, 2*i+1)
            #     imax.imshow(self.x_test[i].reshape(28,28))

            #     predax = plt.subplot(rows, 2*columns, 2*i+2)
            #     correct = False
            #     if np.argmax(y[0]) == np.argmax(self.y_test[i]):
            #         correct = True
            #         num_correct += 1
            #     predax.bar(cat_nums, y[0], color='g' if correct else 'r')
            #     predax.set_xticks(cat_nums)
            #     predax.set_xticklabels(class_names, rotation=60)

            # print(f"Example accuracy: {num_correct/images*100}%")
            # plt.subplots_adjust(hspace=0.4)
            # plt.show()