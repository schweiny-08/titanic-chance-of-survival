from utils import data_prep
from multilayered_perceptron import Multilayered_Perceptron

import numpy as np
import random
import math
import pandas as pd

data = data_prep.read_from_excel_to_array()
np.random.shuffle(data)

# print(len(data))

training_data = data[:(math.ceil(len(data)*0.8))]
testing_data = data[(math.ceil(len(data)*0.8)):]

training_targets, training_inputs = data_prep.separate_target_from_input(training_data)

mlp = Multilayered_Perceptron(
    training_inputs, 
    training_targets,
    num_of_inputs=len(training_inputs[0]),
    num_of_hidden_layers=1,
    num_of_hidden_layer_nodes=300,
    num_of_output_nodes=1,
    eta=0.2,
    mu=0.2
    )

epochs, good_facts, weights = mlp.train()

np.savetxt("final_weights_in_hid.csv", weights[0], delimiter=",")
np.savetxt("final_weights_hid_out.csv", weights[1], delimiter=",")
data_prep.plot_training_graph(epochs, good_facts)

testing_inputs, testing_targets = data_prep.separate_target_from_input(testing_data)

# TODO: Load Weights

print("------------------------------------------------------------------------------")
print("TESTING RUN")
print("------------------------------------------------------------------------------")

mlp.test(testing_inputs, testing_targets, weights)

# mlp.feed_forward()