from FFNN import *
from schedulers import *
from utils import *
from cost_functions import *
from activation_functions import *

from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

seed = np.random.seed(4231)
"""
this script was ran after FFNN_MNIST_adam_hidden, and aims to use the optimal hidden function to optimize the architecture of the layer
"""
inputs = 64
digits = load_digits()

X = digits.data
t_i = digits.target

#One-hot encoding
t = np.zeros((t_i.size, 10))
t[np.arange(t_i.size), t_i] = 1

X_train, X_test, t_train, t_test = train_test_split(X, t)


scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

epochs = 500
folds = 5

hidden_func = sigmoid

eta = 0.001
lam = 0.001

adam = Adam(eta = eta, rho = 0.9, rho2 = 0.99)

batches = 10

#One hidden layer with nodes based on inputs
nodes_in_layer = int((inputs + 1) / 2)
layers_to_try = 4

n_layers_scores, layer = optimize_n_hidden_layers(X_train_scaled, t_train, folds, adam, batches, epochs, lam, nodes_in_layer, layers_to_try, hidden_func = sigmoid, cost_func = CostCrossEntropy,  output_func=softmax)

for i in range(len(n_layers_scores)):
    lab = f"Hidden layer: {layer[i]}"
    plt.plot(n_layers_scores[i]["val_accs"], label = lab)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


#getting the length for next optimization
best_accuracy = 0
best_layer_length = None

for i in range(len(n_layers_scores)):
    current_accuracy = max(n_layers_scores[i]["val_accs"])
    
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_layer_length = len(layer[i])

#Try powers of two in nodes
nodes_to_try = np.power(2, np.arange(7))

#add the number of nodes we initially tested with
nodes_to_try = np.insert(nodes_to_try, 6, nodes_in_layer)

n_nodes_scores, node_layer = optimize_n_nodes(X_train_scaled, t_train, folds, adam, batches, epochs, lam, 4, nodes_to_try, hidden_func = sigmoid, cost_func = CostCrossEntropy,  output_func=softmax)

for i in range(len(n_nodes_scores)):
    lab = f"Hidden layer: {node_layer[i]}"
    plt.plot(n_nodes_scores[i]["val_accs"], label = lab)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()