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
This script was ran second, and checks for the optimal hyperparameters for Adam, but this time using the whole dataset
"""
inputs = 64
digits = load_digits()

X = digits.data
t_i = digits.target

#One-hot encoding
t = np.zeros((t_i.size, 10))
t[np.arange(t_i.size), t_i] = 1

X_train, X_test, t_train, t_test = train_test_split(X, t)

#scaling the data
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#setting parameters to try
etas = np.logspace(-4, -1, 4)
lambdas = np.logspace(-5, -1, 5)
lambdas = np.insert(lambdas, 0, 0)

epochs = 500
folds = 5

rho = 0.9
rho2 = 0.99
momentum = 0.5

batches = 32

#setting the dimensions and FFNN
dims = (inputs, 10)
ffnn = FFNN(dims,output_func=softmax, cost_func=CostCrossEntropy, seed = seed)
scheduler_list = [
    "Adam",
    #"Constant",
    #"Momentum",
    #"Adagrad",
    #"AdagradMomentum",
    #"RMS_prop"
]

best_etas = np.zeros(8)
best_lambdas = np.zeros(8)
i = 0

for s in scheduler_list:
    heatmap, best_eta, best_lambda = ffnn.optimze_params(X_train_scaled, t_train, etas, lambdas, s, batches=batches, epochs=epochs, momentum=momentum, rho=rho, rho2=rho2, folds=folds, X_val = X_test_scaled, t_val= t_test)
    print(f"\n Best eta for {s}: {best_eta}, Best lambda: {best_lambda}")
    ax = sns.heatmap(heatmap, xticklabels=lambdas, yticklabels=etas, annot=True, fmt = ".4f", cmap='viridis_r')
    plt.xlabel("lambda value")
    plt.ylabel("eta value")
    type = "accuracy" if ffnn.classification else "score"
    plt.title(f"{s}, average validation {type} over {folds} folds")
    results_path = f'{s}_results_franke.png'
    plt.savefig(results_path)
    plt.close()
    best_etas[i] = best_eta
    best_lambdas[i] = best_lambda
    i += 1