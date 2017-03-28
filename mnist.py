from __future__ import division, print_function

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def visualise_image(ind,df):

    """
    Visualise an image from a dataframe

    Arguments
    ---------

    ind: integer
        index of the image (row of dataframe) to display

    df: pandas.DataFrame 
        dataframe containing some MNIST data

    Returns
    -------

    ax: matplotlib.axes
        axis object with visualisation of image

    """

    X = df.loc[ind,:][1:].values.reshape((28,28))
    fig,ax = plt.subplots()
    ax.imshow(X,cmap="Greys",interpolation='none')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    return ax

def load_and_preprocess(test=False):

    """
    Get the training data into scikit-friendly format.

    Arguments
    ---------

    test: (=False) bool
        if True, then also return X_test

    Returns
    -------

    X_train: numpy.array
        training data in scikit-friendly format

    y_train: numpy.array
        training labels 

    X_test: numpy.array
        (if test is True) test data in scikit-friendly format

    """


    train = pd.read_csv("data/train.csv")

    y_train = train.label.values
    X_train  = train.drop('label', axis=1).as_matrix()

    if test:
        test = pd.read_csv("data/test.csv")
        X_test = test.as_matrix()
        return X_train/255.,X_test/255.,y_train

    else:
        return X_train/255.,y_train

def fit_linear_svm(X,y):

    """
    Fit a linear SVM to the data. Seems possible to achieve 
    ~ 91 percent accuracy using this simple classifier.

    Arguments
    ---------

    X: numpy.array
        feature matrix 

    y: numpy.array
        training labels

    Returns
    -------

    gs: sklearn.model_selection.GridSearchCV
        cross validation grid-search object trained with 3 fold cross validation
    """

    model = LinearSVC()
    param_grid = {
                  'C': [0.001,0.1,1.,10.] 
                  }
    gs = GridSearchCV(model,param_grid,cv=3,verbose=10,n_jobs=4)
    gs.fit(X,y)

    print("Best average accuracy: ".format(gs.best_score_))
    print("Best value of C: ".format(gs.best_params_['C']))

    return gs





