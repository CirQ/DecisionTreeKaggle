from decision_tree import *


# firstly load the data
print 'Loading data...'
train_X, train_y = fetch_data('./train_clean.csv')
test_X = fetch_data('./test.csv', False)
print 'Data loaded successfully'
print '#'*80


def trial(**kwargs):
    dtc = DecisionTreeClassifier(**kwargs)
    _, acc = evaluate_accuracy(dtc, train_X, train_y, test_X)
    return acc


def main():
    frequency = dict(pd.value_counts(train_y))
    accs = []
    for d in range(1, 30):
        acc = trial(
            criterion='gini',
            max_depth=d,
            min_impurity_decrease=2e-4,
            class_weight=frequency,
            presort=True,
        )
        accs.append(acc)
        print d, 'acc:', acc
    with open('presentation/data.csv', 'w') as w:
        w.write('\n'.join(map(str, accs)))



if __name__ == '__main__':
    main()
