import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate

import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.datasets import load_digits

digit = load_digits()
z  = digit.target
X = digit.data
#Splitting data 4/5 train and 1/5 test, so more data to train than test
X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.4,random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

gd_clf = GradientBoostingClassifier(max_depth=3, n_estimators=100)  
gd_clf.fit(X_train_scaled, z_train)
#Cross validation
accuracy = cross_validate(gd_clf,X_test_scaled,z_test,cv=10)['test_score']
print(accuracy)
print("Test set accuracy with Gradient boosting and scaled data: {:.2f}".format(gd_clf.score(X_test_scaled,z_test)))


