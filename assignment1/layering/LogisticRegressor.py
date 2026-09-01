import numpy as np
import csv, os
from sklearn.linear_model import LogisticRegression

# base directory for all inputs and outputs
DATA_DIR = "/home/iem4723/data"


class LogisticRegressor:

    def __init__(self, trainfilename, testfilename):

        # resolve input files relative to DATA_DIR
        self.X, self.y = self.readData(os.path.join(DATA_DIR, trainfilename))
        self.X_test, self.y_test = self.readData(os.path.join(DATA_DIR, testfilename))

        self.k = self.X.shape[1]
        self.n = self.X.shape[0]

        self.p = self.k + 1

    def readData(self, filename):
        X = []
        y = []

        # Load data set
        with open(filename) as f:
            next(f, None)
            for line in csv.reader(f, delimiter=","):
                X.append(line[:-1])
                y.append(line[-1])

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        return X, y

    def evaluateRegressor(self):

        log_reg, log_score, log_params, log_y_pred = self.logisticRegression(self.X, self.y)

        log_residuals = self.computeResiduals(log_y_pred, self.y)

        # record the regression coefficients (intercept + coefs) to their own CSV file
        self.saveCoefficients(log_params)

        print("\n\n\n\n")

        y_test_pred = self.predictNewObservations(log_reg, self.X_test)

        return y_test_pred - self.y_test

    def logisticRegression(self, X, y):
        reg = LogisticRegression(max_iter=1000).fit(X, y)

        score = reg.score(X, y)

        intercept = reg.intercept_.ravel()
        coefs = reg.coef_.ravel()
        params = np.concatenate((intercept, coefs))

        y_pred = reg.predict(X)

        return reg, score, params, y_pred

    def computeResiduals(self, y_pred, y):
        residuals = y - y_pred
        return residuals

    def predictNewObservations(self, lin_reg, X):
        y_test_pred = lin_reg.predict(X)
        return y_test_pred

    #write the logistic regression coefficients to a CSV in DATA_DIR
    def saveCoefficients(self, params):
        coef_path = os.path.join(DATA_DIR, "logistic_regression_coefficients.csv")
        np.savetxt(coef_path, params, delimiter=",")

alpha = 0.05
logr = LogisticRegressor("wine_quality_train.csv", "wine_quality_test.csv")
log_y_test_pred_err = logr.evaluateRegressor()