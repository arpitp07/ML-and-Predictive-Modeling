# %% [markdown]
# # Machine Learning and Predictive Modeling - Assignment 5
# ### Arpit Parihar
# ### 05/05/2021
# ****
# %% [markdown]
# **Importing modules**
# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from pretty_cm import pretty_plot_confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, auc, plot_roc_curve, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import joblib
import warnings
warnings.filterwarnings('ignore')
# %% [markdown]
# ### 1\. Data Processing
# a) Import the data: You are provided separate .csv files for train and test.
# 
# Train shape: (507, 148)
# Test shape: (168, 148)
# %%
train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

train.shape
test.shape
# %% [markdown]
# b) Remove any rows that have missing data across both sets of data.
# %%
train.dropna(inplace=True)
print(f'Nulls in training set = {train.isnull().sum().sum()}')
test.dropna(inplace=True)
print(f'Nulls in test set = {test.isnull().sum().sum()}')
# %% [markdown]
# c) The target variable (dependent variable) is called "class", make sure to separate this out into a "y_train" and "y_test" and remove from your "X_train" and "X_test". 
# %%
X_train = train.drop(columns='class')
y_train = train['class']
X_test = test.drop(columns='class')
y_test = test['class']
# %% [markdown]
# d) Scale all features / predictors (NOT THE TARGET VARIABLE)
# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %% [markdown]
# **Class to format print statements**
# %%
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# %% [markdown]
# **Function to print classification results**
# %%
def classification_results(act, pred, labels, header=None):
    # labels = ['<=50', '>50']
    print(color.UNDERLINE + color.BOLD + header + ':\n' + color.END)
    print('Confusion Matrix:\n')
    plt.pause(1)
    pretty_plot_confusion_matrix(pd.DataFrame(confusion_matrix(
        act, pred), columns=labels, index=labels), pred_val_axis='x')
    plt.show()
    print('\n' + '-' * 75 + '\n')
    print(classification_report(act, pred, digits=4))
    print('\n' + '-' * 75 + '\n')


# %% [markdown]
# **Functions to plot feature importance and ROC**
# %%
def feature_importance_plot(cols, imp, title=''):
    print(color.UNDERLINE + color.BOLD + title + ' Feature Importance Plot:\n' + color.END)
    fi = sorted(list(imp), reverse=True)[:10]
    cols = [x for _, x in sorted(
        zip(imp, cols), reverse=True)][:10]
    fi_plot = sns.barplot(fi, cols)
    plt.ylabel('Features')
    # plt.xticks(rotation=90)
    plt.xlabel('Importance')
    plt.title(title + ' Feature Importance Plot')
    plt.show(fi_plot);
    print('\n')

def ROC(model, X_test, y_test, title):
    print('\n' + '-' * 75 + '\n')
    print(color.UNDERLINE + color.BOLD + title + ' ROC:\n' + color.END)
    plot_roc_curve(model, X_test, y_test);
    plt.plot([0.01 * x for x in range(101)], [0.01 * x for x in range(101)], linestyle='--');
    plt.title(title + ' ROC');
    plt.pause(2)
    print('\n' + '-' * 75 + '\n')
    

# %% [markdown]
# **Function to fit models - tuned and untuned**
# %%
def model_fit_and_report(X_train, X_test, y_train, y_test, base, title, labels, tuned=False, method='grid', cv=None, best_theshold=False, param_grid=None, save_as=None):
    assert method in ['grid', 'random'], f'Invalid "method" parameter, "{method}". It should be either "grid" or "random".'
    try:
        model = joblib.load(save_as)
        model_best = model.best_estimator_ if tuned else model
    except:
        if tuned:
            if method == 'grid':
                model = GridSearchCV(
                    base, param_grid, cv=cv, refit=True, n_jobs=-1, verbose=0)
            elif method == 'random':
                model = RandomizedSearchCV(
                    base, param_grid, cv=cv, refit=True, n_jobs=-1, verbose=0, n_iter=25)

        else:
            model = base
        model.fit(X_train, y_train)
        model_best = model.best_estimator_ if tuned else model

    if save_as: joblib.dump(model, save_as)

    y_pred = model_best.predict(X_test)
    y_pred_tr = model_best.predict(X_train)

    classification_results(y_test, y_pred, labels,
                           title + ' Test Performance')

    classification_results(y_train, y_pred_tr, labels,
                           title + ' Train Performance')

    return model
# %% [markdown]
# ### 2\. Random Forest Classifier - Base Model:
# %%
labels = sorted(train['class'].unique())
rf_base = model_fit_and_report(
    X_train_scaled, X_test_scaled, y_train, y_test,
    base=RandomForestClassifier(random_state=7),
    title='Base RF',
    labels = labels)
feature_importance_plot(X_train.columns, rf_base.feature_importances_, 'Base RF')
# %% [markdown]
# The model is overfitting, as it achieves 100% accuracy on training data, but the performance drops to 84% on test data. It is fitting to the noise in training data.
# %% [markdown]
# ### 3\. LinearSVM Classifier - Base Model:
# %%
svc_base = model_fit_and_report(
    X_train_scaled, X_test_scaled, y_train, y_test,
    base=LinearSVC(random_state=7),
    title='Base SVC',
    labels = labels)
# %% [markdown]
# There are signs of overfitting, as there's a huge gap between training and test performance, which indicates that the model is fitting to the noise in training data
# %% [markdown]
# ### 4\. Support Vector Machine Classifier + Linear Kernel + Grid Search:
# %%
param_grid = {'C':np.arange(0.01, 10, 0.2).tolist()}
svm_lin = model_fit_and_report(
    X_train_scaled, X_test_scaled, y_train, y_test,
    base=SVC(kernel = 'linear', random_state=7),
    title='Linear SVM',
    labels = labels,
    tuned=True,
    cv=5,
    param_grid=param_grid,
    save_as='svc_linear.pkl')

print(f'Linear SVM best parameters = {svm_lin.best_params_}')
print(f'Linear SVM best estimator = {svm_lin.best_estimator_}')
# %% [markdown]
# There's still slight overfitting, as f1 score drops from 89% for training data to 81% for test data, but it's not as bad as the previous models.
# %% [markdown]
# ### 5\. Support Vector Machine Classifier + Polynomial Kernel + Grid Search:
# %%
param_grid = {'C':np.arange(0.01, 10, 0.2).tolist(), 'degree': [2, 3, 4, 5, 6]}
svm_poly = model_fit_and_report(
    X_train_scaled, X_test_scaled, y_train, y_test,
    base=SVC(kernel = 'poly', random_state=7),
    title='Polynomial SVM',
    labels = labels,
    tuned=True,
    cv=5,
    param_grid=param_grid,
    save_as='svc_polynomial.pkl')

print(f'Polynomial SVM best parameters = {svm_poly.best_params_}')
print(f'Polynomial SVM best estimator = {svm_poly.best_estimator_}')
# %% [markdown]
# This model is overfitting, as the performance on training data is near perfect, but the performance on test data is poorer than most other models fit before.
# %% [markdown]
# ### 6\. Support Vector Machine Classifier + RBF Kernel + Grid Search:
# %%
param_grid = {'C':np.arange(0.01, 10, 0.2).tolist(), 'gamma': [0.01,  0.1, 1, 10, 100]}
svm_rbf = model_fit_and_report(
    X_train_scaled, X_test_scaled, y_train, y_test,
    base=SVC(kernel = 'rbf', random_state=7),
    title='Radial SVM',
    labels = labels,
    tuned=True,
    cv=5,
    param_grid=param_grid,
    save_as='svc_radial.pkl')

print(f'Radial SVM best parameters = {svm_rbf.best_params_}')
print(f'Radial SVM best estimator = {svm_rbf.best_estimator_}')
# %% [markdown]
# Like most other models, this one is overfitting as well. The performance on test data is better than all others, but as the model generalizes poorly, it might not perform this well on more unseen data.
# %% [markdown]
# ### 7\. Conceptual Questions:
# 
# **a) From the models run in steps 2-6, which performs the best based on the Classification Report? Support your reasoning with evidence around your test data.**
# 
# Model 4 - Tuned SVM with linear kernel performs the best. Although its performance is slightly lower than some other models, it's overfitting the least, and is expected to perform consistently on more unseen data, while other models are overfitting a lot, and test performance numbers might change erratically on other test data sets.
# 
# **b) Compare models run for steps 4-6 where different kernels were used. What is the benefit of using a polynomial or rbf kernel over a linear kernel? What could be a downside of using a polynomial or rbf kernel?**
# 
# Model with rbf kernel performs the best out of the bunch on test data, but is overfitting. Polynomial performs the worst, and linear is in between, although it is overfitting the least. Polynomial and RBF kernels are used when the data is not linearly separable, they map the data to a higher dimension function space, where it becomes separable using a hyperplane. The downside of these kernels are their high complexity, which might lead to overfitting, and the higher computational time required.
# 
# **c) Explain the 'C' parameter used in steps 4-6. What does a small C mean versus a large C in sklearn? Why is it important to use the 'C' parameter when fitting a model?**
# 
# C is a regularization parameter which controls how misclassifications are penalized when the model is trained. Smaller C means higher regularization, which means more misclassifications are allowed to avoid overfitting. Higher C value means less regularization, which means very few misclassifications are allowed and the models might overfit.  
# 
# **d) Scaling our input data does not matter much for Random Forest, but it is a critical step for Support Vector Machines. Explain why this is such a critical step. Also, provide an example of a feature from this data set that could cause issues with our SVMs if not scaled.**
# 
# SVMs depend on the distance between the decision boundary and the support vectors to create margins. If data is not scaled, variables with larger magnitudes will have a high influence on the position of the hyperplane used for separation, even when they aren't good predictors. 
# 
# One example from this dataset is the feature **BordLngth_140**, which has the higher magnitude than most other features, and will unproportionately influence the decision boundary if not scaled.
# 
# **e) Describe conceptually what the purpose of a kernel is for Support Vector Machines.**
# 
# A kernel maps data to a higher dimension function space, to separate the data using a hyperplane. The data might not be linearly separable in the current feature space, say 2D for instance, but when transformed to a higher dimensional space, say 3D, using a function or a kernel, it might become separable using a hyperplane.
