from utils import data_prep
from multilayered_perceptron import Multilayered_Perceptron

import numpy as np
import random
import math
import pandas as pd

data = data_prep.read_from_excel_to_array()
np.random.shuffle(data)

training_data = data[:(math.ceil(len(data)*0.8))]
testing_data = data[(math.ceil(len(data)*0.8)):]

training_targets, training_inputs = data_prep.separate_target_from_input(training_data)

mlp = Multilayered_Perceptron(
    training_inputs, 
    training_targets,
    num_of_inputs=len(training_inputs[0]),
    num_of_hidden_layers=1,
    num_of_hidden_layer_nodes=6,
    num_of_output_nodes=1,
    eta=0.2,
    mu=0.2
    )

mlp.train()
# mlp.feed_forward()