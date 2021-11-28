import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import sklearn.preprocessing as pp
from sklearn import metrics

if __name__ == '__main__':

    # load data into numpy array
    # data = np.loadtxt("adult.data", delimiter= ',', dtype=str)
    #
    # # classification data is last column of data
    # y = np.array(data[:,14])
    #
    # # remove last column from data
    # X = np.delete(data,14,1)
    #
    # clf = GaussianNB()
    # # prints for reference
    # print(X[0])
    # print(X[:,0])
    #
    # # convert age to float
    # for x in X[:,0]:
    #     x = float(x)

    df = pd.read_csv('adult.data', names=['age', 'workclass', 'fnlwgt', 'education', 'education-number',
                                          'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                          'capital-loss', 'hours-per-week', 'native-country', 'income'])
    df2 = pd.read_csv('adult.test', names=['age', 'workclass', 'fnlwgt', 'education', 'education-number',
                                          'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                          'capital-loss', 'hours-per-week', 'native-country', 'income'])
    y = df['income']
    X = df.drop(labels='income',axis=1)
    y_test = df['income']
    X_test = df.drop(labels='income', axis=1)
    le = pp.LabelEncoder()
    enc = pp.OneHotEncoder(handle_unknown='ignore')

    X_2 = enc.fit(X)
    X_2 = enc.transform(X).toarray()
    X2_test = enc.fit(X_test)
    X2_test = enc.transform(X_test).toarray()
    gnb = GaussianNB()
    gnb.fit(X_2, y)

    y_pred = gnb.predict(X2_test)
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)



