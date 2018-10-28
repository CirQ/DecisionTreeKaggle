import graphviz
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



def fetch_data(filename, is_train=True):
    data = pd.read_csv(filename)
    if is_train:
        X = data.iloc[:,[0,1,2,3,4,5]]
        y = data.iloc[:,6]
        return X, y
    return data

def write_data(yhat, filename):
    with open('result/{}.csv'.format(filename), 'w') as w:
        w.write('Id,Category')
        for i, y in enumerate(yhat, start=1):
            w.write('\n{},{}'.format(i, int(y)))

def data_transform(model, X, y, Xt):
    model = model.fit(X, y)
    return model.transform(X), model.transform(Xt)



# firstly load the data
print 'Loading data...'
train_X, train_y = fetch_data('./train.csv')
test_X = fetch_data('./test.csv', False)
print 'Data loaded successfully'
print '#'*80



# try to preprocess the features
print 'Try to preprocess data...'

# print 'dimension reduction with primary component analysis'
# pca = PCA(4)
# train_X, test_X = data_transform(pca, train_X, train_y, test_X)

# print 'dimension reduction with linear discriminant analysis'
# lda = LinearDiscriminantAnalysis()
# train_X, test_X = data_transform(lda, train_X, train_y, test_X)

print 'feature selection with lasso'
sfm = SelectFromModel(Lasso(1e-3), max_features=3)
train_X, test_X = data_transform(sfm, train_X, train_y, test_X)

print 'After data preprocessing...'
print train_X
print '#'*80




# frequency = None
frequency = dict(train_y.value_counts())
dtc = DecisionTreeClassifier(
    max_depth=32,
    min_impurity_decrease=1e-4,
    class_weight=frequency,
    presort=True,
)



print 'Start evaluating model...'
score = cross_val_score(dtc, train_X, train_y, scoring='accuracy', cv=10)
print 'The average accuracy:', np.mean(score)
print '#'*80




# try to analyze the generalization issue

label = 'lasso_weight_hyparam'

dtc.fit(train_X, train_y)
pred_y = dtc.predict(test_X)
write_data(pred_y, label)
print 'Predicting data dumpped!'

# dot_data = sklearn.tree.export_graphviz(dtc, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render(label, cleanup=True)
# print 'Graph model drawn!'