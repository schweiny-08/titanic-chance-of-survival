from utils.math_functions import Math_Functions as mf

import numpy as np
import random as rnd

class Multilayered_Perceptron:

    def __init__(self, training_inputs, training_targets, num_of_inputs, num_of_hidden_layers, num_of_hidden_layer_nodes, num_of_output_nodes, eta, mu):
        self.training_inputs = training_inputs
        self.training_targets = training_targets

        self.num_of_hidden_layers = num_of_hidden_layers

        self.weights = mf.create_hidden_layers(len(training_inputs[0]) , num_of_hidden_layers, num_of_hidden_layer_nodes, num_of_output_nodes)

        # Learning Rate
        self.eta = eta
        # Error Threshold
        self.mu = mu

        self.epochs = self.bad_facts = self.good_facts = 0

    def train(self):
        # TODO: Make the proper while loop
        while self.epochs <= 100:
            self.bad_facts = self.good_facts = 0
            self.feed_forward()
            self.epochs+=1

            print(self.epochs)
            print(self.good_facts)
            print(self.bad_facts)


    def feed_forward(self):
        for i in range(len(self.training_inputs)):
            net = []
            out = []
            for j in range(self.num_of_hidden_layers + 1):
                if(j == 0):
                    layer_input = self.training_inputs[i]
                else:
                    layer_input = net[j-1]

                temp_net = mf.activation_function(layer_input, self.weights[j])
                net.append(temp_net)

                temp_out = mf.sigmoid_function(net[j])
                out.append(temp_out)

            error = self.calculate_output_error(out[len(out)-1], self.training_targets[i])

            if(abs(error) > self.mu):
                # TODO: Make backprop
                self.bad_facts+=1
                self.back_propagate()
            else:
                self.good_facts+=1

    def calculate_output_error(self, out, target):
        error = target - out
        return error

    def back_propagate(self):
        return