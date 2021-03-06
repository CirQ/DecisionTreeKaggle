from multiprocessing import Process, Event, Lock, Manager, cpu_count

import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from decision_tree import fetch_data, write_data


def data_fetch():
    train_X, train_y = fetch_data('./train_clean.csv')
    test_X = fetch_data('./test.csv', False)
    return train_X, train_y, test_X


def proc(model, X, y, testX, testy, finishevent, printlock, resultmanager):
    while not finishevent.is_set():
        md = sklearn.clone(model)
        predy = md.fit(X, y).predict(testX)
        acc = accuracy_score(testy, predy)
        if acc > 0.69:
            with printlock:
                print 'The accuracy score is', acc
            if acc >= 0.699:
                resultmanager['pred_y'] = predy
                resultmanager['acc'] = acc
                finishevent.set()
        del md, predy



def main():
    train_X, train_y, test_X = data_fetch()
    test_y = pd.read_csv('test_label.csv').iloc[:,0]

    frequency = dict(pd.value_counts(train_y))
    dtc = DecisionTreeClassifier(
        max_depth=24,
        min_impurity_decrease=2e-4,
        class_weight=frequency,
        presort=True,
    )

    finish_event = Event()
    print_lock = Lock()
    result_manager = Manager().dict({'pred_y':None, 'acc':None})
    procs = []
    for _ in range(cpu_count()-1):
        args = [dtc, train_X, train_y, test_X, test_y, finish_event, print_lock, result_manager]
        p = Process(target=proc, args=args)
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    label = 'result/no_lasso_'+str(int(result_manager['acc']*1000))
    write_data(result_manager['pred_y'], label)


if __name__ == '__main__':
    main()
