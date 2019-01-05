import numpy as np
from sklearn.naive_bayes import GaussianNB


class CreditModel:
    def __init__(self):
        """
        Instantiates the model object, creating class variables if needed.
        """
        self.gnb = GaussianNB()


    def fit(self, X_train, y_train):
        """
        Fits the model based on the given `X_train` and `y_train`.

        You should somehow manipulate and store this data to your model class
        so that you can make predictions on new testing data later on.
        """
        model = self.gnb.fit(X_train, y_train)


    def predict(self, X_test):
        """
        Returns `y_hat`, a prediction for a given `X_test` after fitting.

        You should make use of the data that you stored/computed in the
        fitting phase to make your prediction on this new testing data.
        """
        return self.gnb.predict(X_test)
        # print(preds)


        return np.random.randint(2, size=len(X_test))
