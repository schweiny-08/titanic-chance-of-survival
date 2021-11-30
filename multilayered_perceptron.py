from utils.math_functions import Math_Functions as mf

import numpy as np
import random as rnd
import copy

class Multilayered_Perceptron:

    def __init__(self, training_inputs, training_targets, num_of_inputs, num_of_hidden_layers, num_of_hidden_layer_nodes, num_of_output_nodes, eta, mu):
        self.training_inputs = np.array(training_inputs)
        self.training_targets = np.array(training_targets)

        self.num_of_hidden_layers = num_of_hidden_layers

        self.weights = mf.create_hidden_layers(len(training_inputs[0]) , num_of_hidden_layers, num_of_hidden_layer_nodes, num_of_output_nodes)

        # Learning Rate
        self.eta = eta
        # Error Threshold
        self.mu = mu

        self.epochs = self.bad_facts = self.good_facts = 0

    def train(self):
        # TODO: Make the proper while loop
        while self.good_facts < len(self.training_inputs):
            self.bad_facts = self.good_facts = 0
            self.feed_forward()
            self.epochs+=1

            print("EPOCHS: " + str(self.epochs))
            print("GOOD FACTS: " +  str(self.good_facts))
            print("BAD FACTS: " + str(self.bad_facts))
            print("ACCURACY: " + str((self.good_facts/len(self.training_inputs))*100))

    def feed_forward(self):
        for i in range(len(self.training_inputs)):
            net = []
            out = []
            for j in range(self.num_of_hidden_layers + 1):
                if(j == 0):
                    layer_input = self.training_inputs[i]
                else:
                    layer_input = temp_out

                temp_weights = self.weights[j]

                temp_net = mf.activation_function(layer_input, temp_weights)
                net.append(temp_net)

                temp_out = mf.sigmoid_function(temp_net)
                out.append(temp_out)

            error = self.calculate_output_error(out[len(out)-1], self.training_targets[i])

            if(abs(error) > self.mu):
                self.bad_facts+=1
                # np.savetxt("input.csv", self.training_inputs[i], delimiter=",")
                # np.savetxt("hid_net.csv", net[j-1], delimiter=",")
                # np.savetxt("hid_out.csv", out[j-1], delimiter=",")
                # np.savetxt("out_net.csv", net[j], delimiter=",")
                # np.savetxt("out_out.csv", out[j], delimiter=",")
                # np.savetxt("weights_in_hid.csv", self.weights[j-1], delimiter=",")
                # np.savetxt("weights_hid_out.csv", self.weights[j], delimiter=",")
                self.back_propagate(error=error, temp_weights=self.weights, outputs=out, inputs=self.training_inputs[i])
            else:
                self.good_facts+=1
            
    def back_propagate(self, error, temp_weights, outputs, inputs):
        for i in range(self.num_of_hidden_layers, -1,-1):
            if i == self.num_of_hidden_layers:
                layer_error = copy.deepcopy(error)
            else:
                prev_layer_weights = copy.deepcopy(self.weights[i+1])
                layer_error = self.calculate_hidden_error(prev_layer_delta, prev_layer_weights)

            layer_delta = mf.calculate_delta(outputs[i], layer_error)
            prev_layer_delta = copy.deepcopy(layer_delta)

            if i == 0:
                change_in_weights = self.calculate_change_in_weights(layer_delta, inputs, self.weights[i])
            else:   
                change_in_weights = self.calculate_change_in_weights(layer_delta, outputs[i-1], self.weights[i])

            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] += change_in_weights[j][k]
        return

    def calculate_output_error(self, out, target):
        error = target - out
        return error

    def calculate_hidden_error(self, prev_layer_delta, prev_layer_weights):
        prev_layer_weights = np.transpose(prev_layer_weights)
        error = np.matmul(prev_layer_delta, prev_layer_weights)
        return error

    def calculate_change_in_weights(self, delta, output, weights):
        # to make sure its the same shape
        change_in_weights = copy.deepcopy(weights)
        for x in range(len(change_in_weights)):
            for y in range(len(change_in_weights[x])):
                change_in_weights[x][y] = self.eta * delta[y] * output[x]

        return change_in_weights