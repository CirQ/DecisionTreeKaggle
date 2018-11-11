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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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



# firstly load the data
print 'Loading data...'
train_X, train_y = fetch_data('./train.csv')
test_X = fetch_data('./test.csv', False)
print 'Data loaded successfully'
print '#'*80



print 'Trying outlier detecting and filtering'
lof = LocalOutlierFactor(500, metric='chebyshev', contamination='auto')
train_X, train_y = outlier_filtering(lof, train_X, train_y)
print train_X.shape[0], 'samples remain'
print '#'*80



print 'Print Lasso coefficient'
lasso = Lasso(0.5).fit(train_X, train_y)
print lasso.coef_.tolist()
print 'Plot features histogram'
# display_features(train_X)
print '#'*80


# with open('clean_train.csv', 'w') as w:
#     w.write('Attribute1,Attribute2,Attribute3,Attribute4,Attribute5,Attribute6,Category\n')
#     for i in range(train_X.shape[0]):
#         info = ','.join(map(str, train_X[i]))
#         info += ',{:d}\n'.format(train_y[i])
#         w.write(info)
# exit()


# try to preprocess the features
print 'Try to preprocess data...'

# print 'dimension reduction with primary component analysis'
# pca = PCA(4)
# train_X, test_X = data_transform(pca, train_X, train_y, test_X)

# print 'dimension reduction with linear discriminant analysis'
# lda = LinearDiscriminantAnalysis()
# train_X, test_X = data_transform(lda, train_X, train_y, test_X)

print 'feature selection with lasso'
sfm = SelectFromModel(Lasso(0.5), max_features=3)
train_X, test_X = data_transform(sfm, train_X, train_y, test_X)

print 'After data preprocessing...'
print train_X
print '#'*80



frequency = dict(pd.value_counts(train_y))
dtc = DecisionTreeClassifier(
    max_depth=24,
    min_impurity_decrease=2e-4,
    class_weight=frequency,
    presort=True,
)



# print 'Start brute KFold selecting...'
# _, (train_X, train_y) = brute_sample_select(dtc, train_X, train_y)
# print 'good spliting found'
# print '#'*80



print 'Start evaluating model...'
score = cross_val_score(dtc, train_X, train_y, scoring='accuracy', cv=10)
print 'The average accuracy:', np.mean(score)
print '#'*80




# try to analyze the generalization issue

label = 'lof_hyper_real'

# dtc.fit(train_X, train_y)
# pred_y = dtc.predict(test_X)
# write_data(pred_y, label)
# print 'Predicting data dumpped!'

# dot_data = sklearn.tree.export_graphviz(dtc, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render(label, cleanup=True)
# print 'Graph model drawn!'
