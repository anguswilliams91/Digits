from __future__ import division, print_function

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical


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

def fit_simple_neural_network(X,y):

    """
    Fit a simple neural network to the data. This seems able to achieve 
    ~ 96 percent accuracy.

    Arguments
    ---------

    X: numpy.array
        feature matrix 

    y: numpy.array
        training labels

    Returns
    -------

    data: keras.callbacks.History
        keras history object from training

    model: keras.models.Sequential
        the neural network model 

    """

    y_ohe = to_categorical(y)

    model=Sequential()
    model.add(Dense(32,activation='relu',input_dim=X.shape[1]))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    data=model.fit(X, y_ohe, validation_split = 0.1, epochs=50, batch_size=64)

    return data,model

def fit_convolutional_neural_network(X,y):

    """
    Fit a convolutional neural network to the data. This seems able to achieve ~ 99 percent 
    accuracy.

    Arguments
    ---------

    X: numpy.array
        feature matrix 

    y: numpy.array
        training labels

    Returns
    -------

    data: keras.callbacks.History
        keras history object from training

    model: keras.models.Sequential
        the neural network model 

    """


    y_ohe = to_categorical(y)
    X = X.reshape(X.shape[0],1,28,28)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28),data_format="channels_first"))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )

    data = model.fit(X, y_ohe, validation_split=0.1, epochs=15, batch_size=32)

    return data,model

def main():
    np.random.seed(130)
    X_train,X_test,y_train = load_and_preprocess()
    data,model = fit_convolutional_neural_network(X_train,y_train)
    y_test = model.predict_classes( X_test.reshape(X_test.shape[0],1,28,28) )
    inds = np.arange(1,X_test.shape[0]+1).astype(int)
    submit = pd.DataFrame({'ImageId': inds, 'Label': y_test})
    submit.to_csv("data/submit.csv",index=False)

if __name__ == "__main__":
    main()


