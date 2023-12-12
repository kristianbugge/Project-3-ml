import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.datasets import load_digits

digits = load_digits()
z  = digits.target
X = digits.data
#Splitting data 4/5 train and 1/5 test, so more data to train than test
X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2,random_state=0)

# z = z.reshape(-1,1)
# z_train = z_train.reshape(-1,1) 
# z_test = z_test.reshape(-1,1)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, z_train)
print("Test set accuracy with Logistic Regression: {:.2f}".format(logreg.score(X_test,z_test)))
#now scale the data

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Logistic Regression
logreg.fit(X_train_scaled, z_train)
print("Test set accuracy Logistic Regression with scaled data: {:.2f}".format(logreg.score(X_test_scaled,z_test)))


# Logistic Regression with scaled data
predictions_scaled = logreg.predict(X_test_scaled)

# Create and print confusion matrix for scaled data
conf_matrix_scaled = confusion_matrix(z_test, predictions_scaled)
print("Confusion Matrix (Logistic Regression - Scaled Data):")
print(conf_matrix_scaled)

# Plot confusion matrix heatmap for scaled data
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_scaled, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.title('Confusion Matrix - Scaled Data')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

