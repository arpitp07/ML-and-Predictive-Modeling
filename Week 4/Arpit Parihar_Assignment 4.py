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
from pretty_cm import pretty_plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier as XGBClassifier
import warnings
warnings.filterwarnings('ignore')
# %% [markdown]
# ### 1\. Data Processing
# %%
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'salary']
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
# ### Creating functions to fit models and create performance reports
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

# function to plot feature importance plot
def feature_importance_plot(cols, imp, title=''):
    print(color.UNDERLINE + color.BOLD + title +
          ' Feature Importance Plot:\n' + color.END)
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

# function to fit tuned/untuned models and print performance reports
def model_fit_and_report(X_train, X_test, y_train, y_test, base, title, tuned=False, method='grid', cv=None, param_grid=None, save_as=None):
    assert method in ['grid', 'random'], \
        f'Invalid "method" parameter, "{method}". It should be either "grid" or "random".'
    try:
        model = joblib.load(save_as)
        model_best = model.best_estimator_ if tuned else model
    except:
        if tuned:
            if method == 'grid':
                model = GridSearchCV(
                    base, param_grid, cv=cv,
                    scoring='roc_auc', refit=True, n_jobs=-1, verbose=5)
            elif method == 'random':
                model = RandomizedSearchCV(
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


# %% [markdown]
# ### 2\. Random Forest Classifier - Base Model:
# %%
rf_base = model_fit_and_report(
    X_train, X_test, y_train, y_test,
    base=RandomForestClassifier(random_state=7),
    title='Base RF')
# %% [markdown]
# The base random forest model is overfitting as there is a huge gap in performance between train and test data
# %% [markdown]
# ### 3\. AdaBoost Classifier - GridSearch:
# %%
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.2, 0.4, 0.6, 0.8, 1, 1.2]}

ada_tuned = model_fit_and_report(
    X_train, X_test, y_train, y_test,
    base=AdaBoostClassifier(random_state=7),
    title='AdaBoost',
    tuned=True,
    cv=5,
    param_grid=param_grid,
    save_as='ada_tuned.pkl')
# %% [markdown]
# The train and test performance are comparable for AdaBoost, and the model is not overfitting
# %% [markdown]
# ### 4\. Gradient Boosting Classifier - GridSearch:
# %%
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.4, 0.5, 0.6],
    'max_depth': [1, 2]}

gbm_tuned = model_fit_and_report(
    X_train, X_test, y_train, y_test,
    base=GradientBoostingClassifier(random_state=7),
    title='Gradient Boost',
    tuned=True,
    cv=5,
    param_grid=param_grid,
    save_as='gbm_tuned.pkl')
# %% [markdown]
# The train and test performance are almost the same for gradient boost, and the model is not overfitting
# %% [markdown]
# ### 5\. XGBoost - RandomizedSearchCV
# %%
param_grid = {
    'n_estimators': np.arange(100, 1000 + 50, 50).tolist(),
    'learning_rate': np.arange(0.1, 1.6 + 0.1, 0.1).tolist(),
    'max_depth': [1, 2],
    'gamma': np.arange(0, 5 + 0.25, 0.25).tolist()}

xgb_tuned = model_fit_and_report(
    X_train, X_test, y_train, y_test,
    base=XGBClassifier(random_state=7),
    title='XGBoost',
    tuned=True,
    method='random',
    cv=5,
    param_grid=param_grid,
    save_as='xgb_tuned.pkl')
# %% [markdown]
# The train and test performance are very close for xgboost as well, and the model is not overfitting
# %% [markdown]
# ### 6\. Moving into Conceptual Problems:
# 
# a) What does the alpha parameter represent in AdaBoost? Please refer to chapter 7 of the Hands-On ML book if you are struggling.
# 
# b) In AdaBoost explain how the final predicted class is determined. Be sure to reference the alpha term in your explanation.
# 
# c) In Gradient Boosting, what is the role of the max_depth parameter? Why is it important to tune on this parameter?
# 
# d) In Part (e) of Steps 2-5 you determined the top 5 predictors across each model. Do any predictors show up in the top 5 predictors for all three models? If so, comment on if this predictor makes sense given what you are attempting to predict. (Note: If you don't have any predictors showing up across all 3 predictors, explain one that shows up in 2 of them).
# 
# e) From the models run in steps 2-5, which performs the best based on the Classification Report? Support your reasoning with evidence from your test data and be sure to share the optimal hyperparameters found from your grid search.
# 
# f) For your best performing model, plot out a ROC curve using your test data. Feel free to use sklearn, matplotlib or any other method in python. Describe what the x-axis & y-axis of the ROC curve tell us about a classifier.
