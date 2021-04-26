# %% [markdown]
# # Machine Learning and Predictive Modeling - Assignment 4
# ### Arpit Parihar
# ### 04/26/2021
# ****
# %% [markdown]
# **Importing modules**
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pretty_cm import pretty_plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')
%pylab inline
%config InlineBackend.figure_formats = ['png']
# %% [markdown]
# ### 1\. Data Processing
# %%
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
        'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'salary']
adult_df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None, names=cols, skipinitialspace=True)
adult_df.shape
# %%
adult_df.drop(columns='fnlwgt', inplace=True)
adult_df.salary.replace({'<=50K': 0, '>50K': 1}, inplace=True)

# %%
X = adult_df.drop(columns='salary')
print(f'Shape of X = {X.shape}')
y = adult_df.salary
print(f'Shape of y = {y.shape}')
# %%
X_encoded = pd.get_dummies(X)
print(f'Shape of X_encoded = {X_encoded.shape}')
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, random_state=7, test_size=0.3)
X_train.shape[1] == X_test.shape[1]
# %% [markdown]
# ### 2\. Random Forest Classifier - Base Model:
# %%
rf_base = RandomForestClassifier(random_state=7)
rf_base.fit(X_train, y_train)
# %%
y_pred_base = rf_base.predict(X_test)
y_prob_base = rf_base.predict_proba(X_test)[:, 1]
# %%


class color:    # class to format print statements
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


# function to print classification results
def classification_results(act, pred, prob, header=None):
    labels = ['Not Default', 'Default']
    print(color.UNDERLINE + color.BOLD + header + ':\n' + color.END)
    print('Confusion Matrix:\n')
    plt.pause(1)
    pretty_plot_confusion_matrix(pd.DataFrame(confusion_matrix(
        act, pred), columns=labels, index=labels), pred_val_axis='x')
    plt.show()
    print('\n' + '-' * 75 + '\n')
    print(classification_report(act, pred, digits=4))
    print('\n' + '-' * 75 + '\n')
    print('AUROC: %2.2f%%' % (100 * roc_auc_score(act, prob)))
    print('\n' + '-' * 75 + '\n')
    precision, recall, thresholds = precision_recall_curve(act, prob)
    print('AUPRC: %2.2f%%' % (100 * auc(recall, precision)))
    print('\n')


def feature_importance_plot(cols, imp, title=''):
    print(color.UNDERLINE + color.BOLD + title +
          ' Feature Importance Plot:\n' + color.END)
    fi = sorted(list(imp), reverse=True)[:5]
    cols = [x for _, x in sorted(
        zip(imp, cols), reverse=True)][:5]
    fi_plot = sns.barplot(cols, fi)
    plt.xlabel('Features')
    plt.xticks(rotation=90)
    plt.ylabel('Importance')
    plt.title(title + ' Feature Importance Plot')
    plt.show(fi_plot);
    print('\n')


def model_fit_and_report(X_train, X_test, y_train, y_test, base, title, tuned=False, cv=None, param_grid=None, save_as=None):

    try:
        model = joblib.load(save_as)
        model_best = model.best_estimator_ if tuned else model
    except:
        if tuned:
            model = GridSearchCV(
                base, param_grid, cv=cv,
                scoring='roc_auc', refit=True, n_jobs=-1, verbose=5)
        else:
            model = base
        model.fit(X_train, y_train)
        model_best = model.best_estimator_ if tuned else model

    if save_as: joblib.dump(model, save_as)

    feature_importance_plot(
        X_train.columns, model_best.feature_importances_, title)
    y_pred = model_best.predict(X_test)
    y_prob = model_best.predict_proba(X_test)[:, 1]
    y_pred_tr = model_best.predict(X_train)
    y_prob_tr = model_best.predict_proba(X_train)[:, 1]

    classification_results(y_test, y_pred, y_prob,
                           title + ' Test Performance')
    classification_results(y_train, y_pred_tr, y_prob_tr,
                           title + ' Train Performance')

    return model


# %%
feature_importance_plot(X_train.columns, rf_base.feature_importances_)

# %%
classification_results(y_test, y_pred_base, y_prob_base,
                       'Base RF Test Performance')

# %%
y_pred_base_tr = rf_base.predict(X_train)
y_prob_base_tr = rf_base.predict_proba(X_train)[:, 1]

classification_results(y_train, y_pred_base_tr,
                       y_prob_base_tr, 'Base RF Train Performance')
# %% [markdown]
# The base random forest model is overfitting as there is a huge gap in performance between train and test data
# %% [markdown]
# ### 3\. AdaBoost Classifier - GridSearch:
# %%
try:
    ada_tuned = joblib.load('ada_tuned.pkl')
except:
    # create Random Forest model
    ada_base = AdaBoostClassifier(random_state=7)
    # create a dictionary of parameters
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.2, 0.4, 0.6, 0.8, 1, 1.2]}
    ada_tuned = GridSearchCV(
        ada_base, param_grid, cv=5,
        scoring='roc_auc', refit=True, n_jobs=-1, verbose=5)
    ada_tuned.fit(X_train, y_train)
    joblib.dump(ada_tuned, 'ada_tuned.pkl')

# %%
y_pred_ada = ada_tuned.best_estimator_.predict(X_test)
y_prob_ada = ada_tuned.best_estimator_.predict_proba(X_test)[:, 1]
# %%
feature_importance_plot(
    X_train.columns, ada_tuned.best_estimator_.feature_importances_)
# %%
classification_results(y_test, y_pred_ada, y_prob_ada,
                       'AdaBoost Test Performance')
# %%
# %%
y_pred_ada_tr = ada_tuned.best_estimator_.predict(X_train)
y_prob_ada_tr = ada_tuned.best_estimator_.predict_proba(X_train)[:, 1]

classification_results(y_train, y_pred_ada_tr,
                       y_prob_ada_tr, 'AdaBoost Train Performance')
# %% [markdown]
# The train and test performance are comparable for AdaBoost, and the model is not overfitting
# %%
ada_tuned = model_fit_and_report(X_train, X_test, y_train, y_test, AdaBoostClassifier(
    random_state=7), 'AdaBoost', 5, param_grid, True, 'ada_tuned.pkl')

# %%
rf_base = model_fit_and_report(
    X_train, X_test, y_train, y_test,
    base=RandomForestClassifier(random_state=7),
    title='Base RF',
    tuned=False,
    cv=None,
    param_grid=None,
    save_as=None)

# %%
