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
This script is ran first when optimizing the FFNN for the MNIST dataset. The script gets only the 2's and 5's, with 64 features in order to shorten running times, as
we are trying to find the best scheduler for the case.
"""

inputs = 64
digits = load_digits()

#merge dataset and sort
X_all = np.hstack([digits.data, digits.target.reshape(digits.target.shape[0], 1)])
X_all = X_all[X_all[:, inputs].argsort()]

# get the features of number 2 ONLY
X_twos = X_all[X_all[:, inputs] == 2, :]
X_twos[:, inputs] = 0  # Set the target column to 0
X_twos = resample(X_twos)

# get the features of number 5 ONLY
X_fives = X_all[X_all[:, inputs] == 5, :]
X_fives[:, inputs] = 1
X_fives = resample(X_fives)

#combine the two into a np array
X_all = np.vstack([X_fives, X_twos])

X = X_all[:, :inputs]
t = X_all[:, inputs]

#Splitting into train and test set
X_train, X_test, t_train, t_test = train_test_split(X, t)

t_train = t_train.reshape(t_train.shape[0], 1)
t_test = t_test.reshape(t_test.shape[0], 1)

#Scaling the data with MinMax scaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



#Dimensions and initialization of FFNN
dims = (inputs, 1)
ffnn = FFNN(dims,output_func=sigmoid, cost_func=CostLogReg, seed = seed)

#The learning rates and regularization parameters to try
etas = np.logspace(-4, -1, 4)
lambdas = np.logspace(-5, -1, 5)
lambdas = np.insert(lambdas, 0, 0)

epochs = 500
folds = 5

rho = 0.9
rho2 = 0.99
momentum = 0.5

batches = 32
scheduler_list = [
    "Adam",
    "Constant",
    "Momentum",
    "Adagrad",
    "AdagradMomentum",
    "RMS_prop"
]

best_etas = np.zeros(8)
best_lambdas = np.zeros(8)
i = 0

for s in scheduler_list:
    heatmap, best_eta, best_lambda = ffnn.optimze_params(X_train, t_train, etas, lambdas, s, batches=batches, epochs=epochs, momentum=momentum, rho=rho, rho2=rho2, folds=folds, X_val = X_test, t_val= t_test)
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