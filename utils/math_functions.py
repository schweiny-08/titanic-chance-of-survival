import numpy as np
import random as rnd

class Math_Functions:

    def activation_function(inputs, weights):
        net = np.matmul(inputs, weights)
        return net

    def sigmoid_function(net):
        out = [0 for _ in range(len(net))]
        for i in range(len(out)):
            out[i] = 1/(1 + np.exp(-net[i]))
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

            temp_weights = [[rnd.uniform(-1, 1) for j in range(num_of_layer_output_nodes)] for k in range(num_of_layer_input_nodes)]
            weights.append(temp_weights)
        return weights