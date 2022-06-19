"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=4)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """
        Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        self.weight = None
        self.feature_means = []
        self.feature_stds = []

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """

        result=[]
        for x in X:
            row=[]
            xi=x
            if type(x)==np.ndarray:
                xi=x[0]
            for i in range(1,degree+1):
                row.append(xi**i)
            result.append(row)


        return np.array(result)


    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """
        X=self.polyfeatures(X,self.degree)
        n = len(X)

        self.feature_means=[]
        self.feature_stds=[]

        for j in range(len(X[0])):
            self.feature_means.append(np.mean(X.transpose()[j]))
            self.feature_stds.append(np.std(X.transpose()[j]))



        for i in range(n):
            for j in range(len(X[i])):
                if self.feature_stds[j]!=0:
                    X[i][j]=(X[i][j]-self.feature_means[j])/self.feature_stds[j]

        X_ = np.c_[np.ones([n, 1]), X]

        n, d = X_.shape
        d = d - 1  # remove 1 for the extra column of ones we added to get the original num features

        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(d + 1)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.weight = np.linalg.pinv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        n = len(X)
        X=self.polyfeatures(X,self.degree)
        for i in range(n):
            for j in range(len(X[i])):
                if self.feature_stds[j] != 0:
                    X[i][j] = (X[i][j] - self.feature_means[j]) / self.feature_stds[j]

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X]
        # predict
        return X_.dot(self.weight)




@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    result=0
    for i in range(a.size):
        result+=(a[i]-b[i])**2
    result/=a.size
    return result



@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)


    for i in range(len(Xtrain)):
        model = PolynomialRegression(degree, reg_lambda)
        model.fit(Xtrain[0:(i+1)],Ytrain[0:(i+1)])
        predict_train=model.predict(Xtrain[0:(i+1)])
        errorTrain[i]=mean_squared_error(predict_train,Ytrain)
        predict_test=model.predict(Xtest)
        errorTest[i]=mean_squared_error(predict_test,Ytest)

    # Fill in errorTrain and errorTest arrays

    return [errorTrain,errorTest]
