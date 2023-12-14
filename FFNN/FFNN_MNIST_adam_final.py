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
this is the alst script to run for finding the optimal hidden layer for the given parameters below
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

X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

epochs = 500
folds = 5


eta = 0.001
lam = 0.001

adam = Adam(eta = 0.001, rho = 0.9, rho2 = 0.99)

batches = 10

starting_layer = [64, 64, 64, 64]
try_nodes = [50, 64, 78, 92]

best_accuracy = 0

for i in range(len(starting_layer)):
    curr_scores = np.zeros(len(try_nodes))
    
    for j, nodes in enumerate(try_nodes):
        starting_layer[i] = nodes
        hidden_layer = tuple(starting_layer)
        
        ffnn = FFNN((inputs, *hidden_layer, 10), seed=seed, cost_func=CostCrossEntropy, output_func=softmax, hidden_func=sigmoid)
        
        scores = ffnn.cross_validation(X_train_sc, t_train, folds, adam, batches, epochs, lam)
        acc = np.max(scores["val_accs"])
        
        curr_scores[j] = acc
        print(f"Current hidden layer {hidden_layer}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_nodes = nodes
            
    starting_layer[i] = best_nodes



print("Best Accuracy:", best_accuracy)
print("Best Hidden Layer Configuration:", starting_layer)
