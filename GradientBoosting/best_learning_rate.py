import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate
import scikitplot as skplt

import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

#---------------------- Setting Up the Data-----------------------
np.random.seed(4231)
digits = load_digits()
z  = digits.target
X = digits.data
#Splitting data 4/5 train and 1/5 test, so more data to train than test
X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2,random_state=0)

n_estimators = 100
max_depth = 3
learning_rate = 0.3

#----------------------Using sklearn GradientBoostingClassifier-----------------

learning_rates = np.arange(0.05,1.05,0.05)

accuracy_scores = np.zeros(learning_rates.size)
i = 0

for learning_rate in learning_rates:
    # Unscaled Data
    gd_clf = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)  
    gd_clf.fit(X_train, z_train)
    z_pred = gd_clf.predict(X_test)

    # Accuracy Test
    accuracy_scores[i] = gd_clf.score(X_test,z_test)
    i+=1

plt.figure(figsize=(16,9))
plt.plot(learning_rates,accuracy_scores, label = "Accuracy Score")
plt.xlabel("Learning Rate ", fontsize = 16)
plt.ylabel("Accuracy Score", fontsize = 16)
plt.title("Accuracy Score on different learning rates", fontsize = 16)
plt.xticks(learning_rates, fontsize = 14)
plt.yticks(fontsize = 14)
plt.ylim(0.90,1)
plt.legend(fontsize = 16)
plt.show()
