from decision_tree import *


# firstly load the data
print 'Loading data...'
train_X, train_y = fetch_data('./train_clean.csv')
test_X = fetch_data('./test.csv', False)
print 'Data loaded successfully'
print '#'*80



# print 'Try to clean data...'
#
# print 'Trying outlier detecting and filtering'
# lof = LocalOutlierFactor(100, metric='chebyshev', contamination='auto')
# train_X, train_y = outlier_filtering(lof, train_X, train_y)
# print train_X.shape[0], 'samples remain'
# print '#'*80
#
# dump_data = pd.DataFrame(train_X, columns=['Attribute%d'%i for i in range(1,7)])
# dump_data['Category'] = pd.Series(train_y)
# dump_data.to_csv('train_clean.csv', index=False)



lasso = Lasso(0.5).fit(train_X, train_y)
print 'Print Lasso coefficient'
print lasso.coef_.tolist()
# print 'Plot features histogram'
# display_features(train_X)
print '#'*80


# try to preprocess the features
print 'Try to preprocess data...'

# print 'dimension reduction with primary component analysis'
# pca = PCA(4)
# train_X, test_X = data_transform(pca, train_X, train_y, test_X)
#
# print 'dimension reduction with linear discriminant analysis'
# lda = LinearDiscriminantAnalysis()
# train_X, test_X = data_transform(lda, train_X, train_y, test_X)

# print 'feature selection with lasso'
# sfm = SelectFromModel(Lasso(0.5), max_features=4)
# train_X, test_X = data_transform(sfm, train_X, train_y, test_X)

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
pred_y, acc = evaluate_accuracy(dtc, train_X, train_y, test_X)
print 'The accuracy is', acc
print '#'*80




# try to analyze the generalization issue

label = 'real_evaluate'

# write_data(pred_y, label)
# print 'Predicting data dumpped!'

# dot_data = sklearn.tree.export_graphviz(dtc, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render(label, cleanup=True)
# print 'Graph model drawn!'
