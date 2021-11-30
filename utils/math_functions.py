import numpy as np
import random as rnd

class Math_Functions:

    def activation_function(inputs, weights):
        net = np.dot(inputs, weights)
        return net

    def sigmoid_function(net):
        out = net
        out = 1/(1 + np.exp(-net))
        return out

    def create_hidden_layers(num_of_input_nodes, num_of_hidden_layers, num_of_hidden_layer_nodes, num_of_output_nodes):
        weights = []
        for i in range(num_of_hidden_layers+1):
            if(i == 0):
                num_of_layer_input_nodes = num_of_input_nodes
                num_of_layer_output_nodes = num_of_hidden_layer_nodes
            elif(i >= num_of_hidden_layers):
                num_of_layer_input_nodes = num_of_hidden_layer_nodes
                num_of_layer_output_nodes = num_of_output_nodes
            else:
                num_of_layer_input_nodes = num_of_layer_output_nodes = num_of_hidden_layer_nodes

            temp_weights = [[ 0.0 for j in range(num_of_layer_output_nodes)] for k in range(num_of_layer_input_nodes)]
            
            for i in range(len(temp_weights)):
                for j in range(len(temp_weights[i])):
                    temp_weights[i][j] = rnd.uniform(-1, 1)

            weights.append(temp_weights)
        return weights

    def calculate_delta(output, error):
        deriv = Math_Functions.sigmoid_derivative(output)
        delta = deriv * error
        return delta

    def sigmoid_derivative(output):
        temp = 1 - output
        deriv = output * temp
        return deriv