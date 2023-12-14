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
this script can be ran with given parameters found from the other scripts to plot a 10x10 confusion matrix of the resulting accuracy
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


eta = 0.001
lam = 0.001

adam = Adam(eta = 0.001, rho = 0.9, rho2 = 0.99)

batches = 32

hidden_layer = (92, 64, 64, 64)

ffnn = FFNN((inputs, *hidden_layer, 10), seed=seed, cost_func=CostCrossEntropy, output_func=softmax, hidden_func=sigmoid)

scores = ffnn.cross_validation(X_train_scaled, t_train, folds, adam, batches, epochs, lam, X_val = X_test_scaled, t_val = t_test)
sns.heatmap(scores["confusion_matrix"], annot=True, fmt = ".3%",  cmap='Greens')
plt.title("Confusion matrix of MNIST dataset after ffnn fitting")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()