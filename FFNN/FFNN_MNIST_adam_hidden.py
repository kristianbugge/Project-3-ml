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
This script was ran third and uses the parameters from FFNN_MNIST_adam.py to find the best hidden layer function.
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


etas = np.logspace(-4, -2, 3)
lambdas = np.logspace(-5, -2, 4)

rho = 0.9
rho2 = 0.99
momentum = 0.5

epochs = 500
folds = 5

batches = 10

#One hidden layer with nodes based on inputs
hidden_layer = int(inputs + 1 / 2)

scheduler = "Adam"
hidden_funcs = [sigmoid, RELU, LRELU]

for func in hidden_funcs:
    ffnn = FFNN(dimensions=(inputs, hidden_layer, 10), hidden_func=func, seed=4231, output_func= softmax, cost_func= CostCrossEntropy)
    heatmap, best_eta, best_lambda = ffnn.optimze_params(X_train_scaled, t_train, etas, lambdas, scheduler, batches = batches, epochs = epochs, momentum=momentum, rho=rho, rho2=rho2, folds = folds, X_val= X_test_scaled, t_val = t_test)
    print(f"\n Best eta for {scheduler} with {func}: {best_eta}, Best lambda: {best_lambda}")
    ax = sns.heatmap(heatmap, xticklabels=lambdas, yticklabels=etas, annot=True, fmt = ".4f", cmap='viridis_r')
    plt.xlabel("lambda value")
    plt.ylabel("eta value")
    plt.title(f"{scheduler}, average validation error over {folds} folds using activation function: {func}")
    plt.show()