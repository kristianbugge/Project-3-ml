import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt

import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.datasets import load_digits

#---------------------- Setting Up the Data-----------------------

digits = load_digits()
z  = digits.target
X = digits.data
#Splitting data 4/5 train and 1/5 test, so more data to train than test
X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2,random_state=0)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

#----------------------Using sklearn LogisticRegression-----------------

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, z_train)
print("Test set accuracy with Logistic Regression: {:.2f}".format(logreg.score(X_test,z_test)))

# Logistic Regression with scaled data
z_pred = logreg.predict(X_test)

# Plot confusion matrix heatmap for scaled data
conf_matrix = confusion_matrix(z_test, z_pred)
plt.figure(figsize=(16, 16))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=digits.target_names, yticklabels=digits.target_names, annot_kws={"size": 24})
plt.title('Confusion Matrix on Scikit-Learn Logistic Regression, unscaled Data', fontsize = 28)
plt.xlabel('Predicted', fontsize = 24)
plt.ylabel('True', fontsize = 24)
results_path = f'scikitlearn_unscaled_CM_LogisticRegression.png'
plt.savefig(results_path)
plt.close()

# ROC Curve
skplt.metrics.plot_confusion_matrix(z_test, z_pred, normalize=True, title='Confusion Matrix Scikit-Learn Logistic Regression (probas), unscaled data')
results_path = f'scikitlearn_unscaled_CM%_LogisticRegression.png'
plt.savefig(results_path)
plt.close()
z_probas = logreg.predict_proba(X_test)
skplt.metrics.plot_roc(z_test, z_probas, title='ROC Curve Scikit-Learn Logistic Regression, unscaled data', figsize=(16,9), text_fontsize=16, title_fontsize=20)
results_path = f'scikitlearn_unscaled_ROC_LogisticRegression.png'
plt.savefig(results_path)
plt.close()

# Now scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logreg.fit(X_train_scaled, z_train)
print("Test set accuracy Logistic Regression with scaled data: {:.2f}".format(logreg.score(X_test_scaled,z_test)))

# Logistic Regression with scaled data
z_pred_scaled = logreg.predict(X_test_scaled)

# Plot confusion matrix heatmap for scaled data
conf_matrix_scaled = confusion_matrix(z_test, z_pred_scaled)
plt.figure(figsize=(16, 12))
sns.heatmap(conf_matrix_scaled, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=digits.target_names, yticklabels=digits.target_names, annot_kws={"size": 24})
plt.title('Confusion Matrix on Scikit-Learn Logistic Regression, scaled Data', fontsize = 28)
plt.xlabel('Predicted', fontsize = 24)
plt.ylabel('True', fontsize = 24)
results_path = f'scikitlearn_scaled_CM_LogisticRegression.png'
plt.savefig(results_path)
plt.close()

# ROC Curve
skplt.metrics.plot_confusion_matrix(z_test, z_pred_scaled, normalize=True, title='Confusion Matrix Scikit-Learn Logistic Regression (probas), scaled data')
results_path = f'scikitlearn_scaled_CM%_LogisticRegression.png'
plt.savefig(results_path)
plt.close()
z_probas = logreg.predict_proba(X_test_scaled)
skplt.metrics.plot_roc(z_test, z_probas, title='ROC Curve Scikit-Learn Logistic Regression, scaled data', figsize=(16,9), text_fontsize=16, title_fontsize=20)
results_path = f'scikitlearn_scaled_ROC_LogisticRegression.png'
plt.savefig(results_path)
plt.close()