import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

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
    y = df['income']
    X = df.drop(labels='income',axis=1)
    print(type(df['workclass'][0]))

