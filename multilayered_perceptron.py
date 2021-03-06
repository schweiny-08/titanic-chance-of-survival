from utils.math_functions import Math_Functions as mf

import numpy as np
import random as rnd
import copy

class Multilayered_Perceptron:

    def __init__(self, training_inputs, training_targets, num_of_inputs, num_of_hidden_layers, num_of_hidden_layer_nodes, num_of_output_nodes, eta, mu):
        self.inputs = np.array(training_inputs)
        self.targets = np.array(training_targets)

        self.num_of_hidden_layers = num_of_hidden_layers

        self.weights = mf.create_hidden_layers(len(training_inputs[0]) , num_of_hidden_layers, num_of_hidden_layer_nodes, num_of_output_nodes)

        # Learning Rate
        self.eta = eta
        # Error Threshold
        self.mu = mu

        self.epochs = self.bad_facts = self.good_facts = 0

    def train(self):
        good_facts_final = []
        # Trains for 10k epochs
        while self.epochs < 10000:
            self.bad_facts = self.good_facts = 0
            self.feed_forward(is_training=True)
            self.epochs+=1
            good_facts_final.append(self.good_facts)

            print("EPOCHS: " + str(self.epochs))
            print("GOOD FACTS: " +  str(self.good_facts))
            print("BAD FACTS: " + str(self.bad_facts))
            print("ACCURACY: " + str((self.good_facts/len(self.inputs))*100))
        
        return self.epochs, good_facts_final, self.weights

    def test(self, testing_inputs, testing_targets, weights):
        self.bad_facts = self.good_facts = 0
        self.weights = weights
        self.inputs = testing_inputs
        self.targets = testing_targets
        self.feed_forward(is_training=False)

        print("FINAL GOOD FACTS: " +  str(self.good_facts))
        print("FINAL BAD FACTS: " + str(self.bad_facts))
        print("FINAL ACCURACY: " + str((self.good_facts/len(self.inputs))*100))


    def feed_forward(self, is_training=True):
        for i in range(len(self.inputs)):
            net = []
            out = []
            for j in range(self.num_of_hidden_layers + 1):
                if(j == 0):
                    layer_input = self.inputs[i]
                else:
                    layer_input = temp_out

                temp_weights = copy.deepcopy(self.weights[j])

                temp_net = mf.activation_function(layer_input, temp_weights)
                net.append(temp_net)

                temp_out = mf.sigmoid_function(temp_net)
                out.append(temp_out)

            error = self.calculate_output_error(out[len(out)-1], self.targets[i])

            if(abs(error) > self.mu):
                self.bad_facts+=1
                if is_training:
                    self.back_propagate(error=error, outputs=out, inputs=self.inputs[i])
            else:
                self.good_facts+=1
            
    def back_propagate(self, error, outputs, inputs):
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