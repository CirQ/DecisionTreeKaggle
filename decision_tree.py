import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier



def fetch_data(filename, is_train=True):
    data = pd.read_csv(filename)
    if is_train:
        X = data.iloc[:,[0,1,2,3,4,5]]
        y = data.iloc[:,6]
        return X.values, y.values
    return data.values

def outlier_filtering(fltr, X, y):
    out = fltr.fit_predict(X)
    return X[out==1], y[out==1]

def data_transform(model, X, y, Xt):
    model = model.fit(X, y)
    return model.transform(X), model.transform(Xt)

def write_data(yhat, filename):
    with open('result/{}.csv'.format(filename), 'w') as w:
        w.write('Id,Category')
        for i, y in enumerate(yhat, start=1):
            w.write('\n{},{}'.format(i, int(y)))

def display_features(X):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,7))
    names = ['Attribute1', 'Attribute2', 'Attribute3',
             'Attribute4', 'Attribute5', 'Attribute6']
    for i, column in enumerate(X.T):
        x, y = i / 3, i % 3
        axes[x,y].hist(column)
        axes[x,y].xaxis.set_major_locator(tic.MaxNLocator(4))
        axes[x,y].set_title('#'+names[i])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.show()

def brute_sample_select(model, X, y, threshold=0.7):
    kf = KFold(n_splits=5, shuffle=True)
    while True:
        scoring = {}
        for train, validate in kf.split(X, y):
            X_train, y_train = X[train], y[train]
            X_validate, y_validate = X[validate], y[validate]
            model = model.fit(X_train, y_train)
            y_pred = model.predict(X_validate)
            acc = accuracy_score(y_validate, y_pred)
            scoring[acc] = (X_train, y_train)
        max_score = max(scoring)
        print 'result, socre =', max_score
        if max_score > threshold:
            return max_score, scoring[max_score]

def evaluate_accuracy(model, X, y, X_test=None):
    if X_test is None:
        accs = cross_val_score(model, X, y, scoring='accuracy', cv=10)
        return None, np.mean(accs)
    else:
        y_real = pd.read_csv('test_label.csv').iloc[:,0]
        model.fit(X, y)
        y_pred = model.predict(X_test)
        return y_pred, accuracy_score(y_real, y_pred)

def display_clean_diff():
    # TODO: show two subplots
    pass
