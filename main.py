import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import sklearn.preprocessing as pp
from sklearn import metrics
import category_encoders as ce

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
    y_test = df2['income']
    X_test = df2.drop(labels='income', axis=1)
    #X = X.drop([0])
    #X_test = X_test.drop([0])
    le = pp.LabelEncoder()
    # enc = pp.OneHotEncoder(handle_unknown='ignore')
    # enc2 = pp.OneHotEncoder(handle_unknown='ignore')

    enc = ce.count.CountEncoder()
    enc2 = ce.count.CountEncoder()

    X = enc.fit_transform(X,y)
    X_test = enc2.fit_transform(X_test, y_test)
    y = le.fit_transform(y)
    y_test = le.fit_transform(y_test)

    #X_2 = enc.fit(X)
    #X_2 = enc.fit_transform(X).toarray()
    #X2_test = enc2.fit(X_test)
    #X2_test = enc2.fit_transform(X_test).toarray()
    gnb = GaussianNB()
    #gnb.fit(X_2, y)

    gnb.fit(X, y)

    y_pred = gnb.predict(X_test)
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)



