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

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_estimators = 100
max_depth = 3
learning_rate = 0.3

#----------------------Using sklearn GradientBoostingClassifier-----------------

# Unscaled Data
gd_clf = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)  
gd_clf.fit(X_train, z_train)
z_pred = gd_clf.predict(X_test)

# # Cross validation
# accuracy = cross_validate(gd_clf,X_test,z_test,cv=10)['test_score']
# print(accuracy)

# Accuracy Test
print("ScikitLearn Test set accuracy with Gradient boosting and unscaled data: {:.2f}".format(gd_clf.score(X_test,z_test)))
conf_matrix = confusion_matrix(z_test, z_pred)

# Plot confusion matrix heatmap for unscaled data
plt.figure(figsize=(16, 16))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=digits.target_names, yticklabels=digits.target_names, annot_kws={"size": 24})
plt.title('Confusion Matrix Scikit-Learn Gradient Boosting, unscaled data', fontsize = 28)
plt.xlabel('Predicted', fontsize = 24)
plt.ylabel('True for class', fontsize = 24)
results_path = f'scikitlearn_unscaled_CM_Boosting.png'
plt.savefig(results_path)
plt.close()

# ROC Curve
skplt.metrics.plot_confusion_matrix(z_test, z_pred, normalize=True, title='Confusion Matrix Scikit-Learn GBC (proba), unscaled', figsize=(16,16), text_fontsize=24, title_fontsize=28)
cax = plt.gcf().axes[-1]
cax.tick_params(axis='y', labelsize=20) 
results_path = f'scikitlearn_unscaled_CM%_Boosting.png'
plt.savefig(results_path)
plt.close()
z_probas = gd_clf.predict_proba(X_test)
skplt.metrics.plot_roc(z_test, z_probas, title='ROC Curve Scikit-Learn Gradient Boosting, unscaled data', figsize=(16,9), text_fontsize=16, title_fontsize=20)
results_path = f'scikitlearn_unscaled_ROC_Boosting.png'
plt.savefig(results_path)
plt.close()
# skplt.metrics.plot_cumulative_gain(z_test, z_probas)
# results_path = f'scikitlearn_unscaled_CG_Boosting.png'
# plt.savefig(results_path)
# plt.close()

# Scaled Data
gd_clf = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)  
gd_clf.fit(X_train_scaled, z_train)
z_pred = gd_clf.predict(X_test_scaled)
# # Cross validation
# accuracy = cross_validate(gd_clf,X_test_scaled,z_test,cv=10)['test_score']
# print(accuracy)

# Accuracy Test
print("ScikitLearn Test set accuracy with Gradient boosting and scaled data: {:.2f}".format(gd_clf.score(X_test_scaled,z_test)))
conf_matrix = confusion_matrix(z_test, gd_clf.predict(X_test_scaled))

# Plot confusion matrix heatmap for scaled data
plt.figure(figsize=(16, 16))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=digits.target_names, yticklabels=digits.target_names, annot_kws={"size": 24})
plt.title('Confusion Matrix Scikit-Learn Gradient Boosting, scaled data')
plt.xlabel('Predicted', fontsize = 24)
plt.ylabel('True for class', fontsize = 24)
results_path = f'scikitlearn_scaled_CM_Boosting.png'
plt.savefig(results_path)
plt.close()

# ROC Curve
skplt.metrics.plot_confusion_matrix(z_test, z_pred, normalize=True, title='Confusion Matrix Scikit-Learn GBC (proba), scaled', figsize=(16,16), text_fontsize=24, title_fontsize=28)
cax = plt.gcf().axes[-1]
cax.tick_params(axis='y', labelsize=20) 
results_path = f'scikitlearn_scaled_CM%_Boosting.png'
plt.savefig(results_path)
plt.close()
z_probas = gd_clf.predict_proba(X_test_scaled)
skplt.metrics.plot_roc(z_test, z_probas, title='ROC Curve Scikit-Learn Gradient Boosting, scaled data', figsize=(16,9), text_fontsize=16, title_fontsize=20)
results_path = f'scikitlearn_scaled_ROC_Boosting.png'
plt.savefig(results_path)
plt.close()
# skplt.metrics.plot_cumulative_gain(z_test, z_probas)
# results_path = f'scikitlearn_scaled_CG_Boosting.png'
# plt.savefig(results_path)
# plt.close()

#----------------------Without using sklearn GradientBoostingClassifier-----------------

from GradientBoostingClassifier import *
    
gbcfs = GradientBoostingClassifierFromScratch(n_estimators=n_estimators, 
                                              learning_rate=learning_rate, 
                                              max_depth=max_depth)
# Unscaled Data
gbcfs.fit(X_train, z_train)
z_pred = gbcfs.predict(X_test)

# Accuracy Test
print("Handmade Test set accuracy with Gradient boosting and unscaled data: {:.2f}".format(accuracy_score(z_test, gbcfs.predict(X_test))))
conf_matrix = confusion_matrix(z_test, z_pred)

# Plot confusion matrix heatmap for scaled data
plt.figure(figsize=(16, 16))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=digits.target_names, yticklabels=digits.target_names, annot_kws={"size": 24})
plt.title('Confusion Matrix on handmade, unscaled data')
plt.xlabel('Predicted', fontsize = 24)
plt.ylabel('True for class', fontsize = 24)
results_path = f'handmade_unscaled_CM_Boosting.png'
plt.savefig(results_path)
plt.close()

# ROC Curve
skplt.metrics.plot_confusion_matrix(z_test, z_pred, normalize=True, title='Confusion Matrix on handmade GBC (proba), unscaled', figsize=(16,16), text_fontsize=24, title_fontsize=28)
cax = plt.gcf().axes[-1]
cax.tick_params(axis='y', labelsize=20) 
results_path = f'handmade_unscaled_CM%_Boosting.png'
plt.savefig(results_path)
plt.close()
z_probas = gbcfs.predict_proba(X_test)
skplt.metrics.plot_roc(z_test, z_probas, title='ROC Curve on handmade Gradient Boosting, unscaled data', figsize=(16,9), text_fontsize=16, title_fontsize=20)
results_path = f'handmade_unscaled_ROC_Boosting.png'
plt.savefig(results_path)
plt.close()
# skplt.metrics.plot_cumulative_gain(z_test, z_probas)
# results_path = f'handmade_unscaled_CG_Boosting.png'
# plt.savefig(results_path)
# plt.close()

# Scaled Data
gbcfs.fit(X_train_scaled, z_train)
z_pred = gbcfs.predict(X_test_scaled)

# Accuracy Test
print("Handmade Test set accuracy with Gradient boosting and scaled data: {:.2f}".format(accuracy_score(z_test, gbcfs.predict(X_test_scaled))))
conf_matrix = confusion_matrix(z_test, z_pred)

# Plot confusion matrix heatmap for scaled data
plt.figure(figsize=(16, 16))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=digits.target_names, yticklabels=digits.target_names, annot_kws={"size": 24})
plt.title('Confusion Matrix on Handmade Gradient Boosting, scaled data')
plt.xlabel('Predicted', fontsize = 24)
plt.ylabel('True for class', fontsize = 24)
results_path = f'handmade_scaled_CM_Boosting.png'
plt.savefig(results_path)
plt.close()

# ROC Curve
skplt.metrics.plot_confusion_matrix(z_test, z_pred, normalize=True, title='Confusion Matrix on handmade GBC (proba), scaled', figsize=(16,16), text_fontsize=24, title_fontsize=28)
cax = plt.gcf().axes[-1]
cax.tick_params(axis='y', labelsize=20) 
results_path = f'handmade_scaled_CM%_Boosting.png'
plt.savefig(results_path)
plt.close()
z_probas = gbcfs.predict_proba(X_test_scaled)
skplt.metrics.plot_roc(z_test, z_probas, title='ROC Curve on handmade Gradient Boosting, scaled data', figsize=(16,9), text_fontsize=16, title_fontsize=20)
results_path = f'handmade_scaled_ROC_Boosting.png'
plt.savefig(results_path)
plt.close()
# skplt.metrics.plot_cumulative_gain(z_test, z_probas)
# results_path = f'handmade_scaled_CG_Boosting.png'
# plt.savefig(results_path)
# plt.close()
