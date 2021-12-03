import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

def read_from_excel_to_array():
    data = pd.read_excel(r'.\data\normalized_data.xlsx', header=0)
    data_array = data.to_numpy()
    return data_array
    
def separate_target_from_input(data_array):
    targets = [element[0] for element in data_array]
    inputs = [element[1:] for element in data_array]
    for i in range(len(inputs)):
        inputs[i] = inputs[i]
    # inputs = normalize_inputs(inputs)
    return targets, inputs

def normalize_inputs(inputs):
    inputs = preprocessing.normalize(inputs)
    return inputs

def plot_training_graph(epochs, good_facts):
    total_epochs = [ i+1 for i in range(epochs)]

    plt.plot(total_epochs, good_facts)
    plt.xlabel("Epochs")
    plt.ylabel("Good Facts")
    plt.title("Training Results")
    plt.savefig("TrainingGraph.png")