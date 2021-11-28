import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':

    # load data into numpy array
    data = np.loadtxt("adult.data", delimiter= ',', dtype=str)

    # classification data is last column of data
    y = np.array(data[:,14])

    # remove last column from data
    X = np.delete(data,14,1)

    clf = GaussianNB()
    # prints for reference
    print(X[0])
    print(X[:,0])

    # convert age to float
    for x in X[:,0]:
        x = float(x)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X)
    print(enc.categories_)
   # clf.fit(X,y)

