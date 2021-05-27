"""
experiment.py

Apply classification algorithms to a dataset and evaluate the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from preprocess import preprocess, load_dataset
import pickle

SAVEDIR = 'demo'
DATADIR = 'data/muraro'
DATASET = 'muraro'
AVERAGE = None

def evaluate_classifier(clf, X_train, X_test, y_train, y_test, average='macro'):
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(y_test, prediction,
                                                                         average=average,
                                                                         zero_division='warn')
    return precision, recall, f1

def main():

    # NOTE: This is only tentative. Let's think about what results do we want first.

    # Load data and preprocess
    data = load_dataset(DATADIR, DATASET)
    X, y = preprocess(data)
    num_labels = len(np.unique(y))

    # Train/Test splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)

    eval_log = {}

    # SVM classifier with linear kernel
    clf = svm.SVC(kernel='linear')
    precision, recall, f1 = evaluate_classifier(clf, X_train, X_test, y_train, y_test, average=AVERAGE)
    eval_log['svm-linear'] = (precision, recall, f1)

    # SVM classifier with polynomial kernel
    clf = svm.SVC(kernel='poly', degree=3)
    precision, recall, f1 = evaluate_classifier(clf, X_train, X_test, y_train, y_test, average=AVERAGE)
    eval_log['svm-poly'] = (precision, recall, f1)

    # SVM classifier with RBF kernel
    clf = svm.SVC(kernel='rbf')
    precision, recall, f1 = evaluate_classifier(clf, X_train, X_test, y_train, y_test, average=AVERAGE)
    eval_log['svm-rbf'] = (precision, recall, f1)

    # SVM classifier with sigmoid kernel
    clf = svm.SVC(kernel='sigmoid')
    precision, recall, f1 = evaluate_classifier(clf, X_train, X_test, y_train, y_test, average=AVERAGE)
    eval_log['svm-sigmoid'] = (precision, recall, f1)

    # kNN classifier with Euclidean distance
    clf = KNeighborsClassifier(n_neighbors=num_labels, metric='euclidean')
    precision, recall, f1 = evaluate_classifier(clf, X_train, X_test, y_train, y_test, average=AVERAGE)
    eval_log['knn-euclidean'] = (precision, recall, f1)

    # kNN classifier with Manhattan distance
    clf = KNeighborsClassifier(n_neighbors=num_labels, metric='manhattan')
    precision, recall, f1 = evaluate_classifier(clf, X_train, X_test, y_train, y_test, average=AVERAGE)
    eval_log['knn-manhattan'] = (precision, recall, f1)

    # Multi-layer perceptron classifier
    clf = MLPClassifier()
    precision, recall, f1 = evaluate_classifier(clf, X_train, X_test, y_train, y_test, average=AVERAGE)
    eval_log['mlp'] = (precision, recall, f1)

    pickle.dump(eval_log, open('{}.p'.format(SAVEDIR), 'wb'))

