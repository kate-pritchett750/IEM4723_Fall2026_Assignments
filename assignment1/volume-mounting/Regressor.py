import numpy as np
import time, os, json

#some constants being used for analytics
DATASET_SIZE=300
RECORDS=250
FEATURES=10
LABELS=10
COLUMN_SIZE=FEATURES+LABELS
SLEEP_SECS = 15


'''
function to process the raw data
'''
def processStreamingData(raw_data):

    start = np.random.randint(DATASET_SIZE-RECORDS,size=1)[0]
    end = start + RECORDS
    features = raw_data[start:end, 0:FEATURES]
    labels = raw_data[start:end, FEATURES:COLUMN_SIZE]

    test_features = np.reshape(raw_data[end, 0:FEATURES],(FEATURES,1))
    test_labels = np.reshape(raw_data[end, FEATURES:COLUMN_SIZE],(FEATURES,1))
    return features, labels, test_features, test_labels


#function to store output of regression computation
def processRegressionOutput(regression_coefficients):
    np.savetxt("/home/iem4723/data/regression_coefficients.csv",regression_coefficients,delimiter=',')


#read stream.csv located in /tmp/stream.csv
raw_data = np.loadtxt("/home/iem4723/data/stream.csv", delimiter=',', encoding='utf-8-sig')

#process streaming data
features,labels, test_features, test_labels = processStreamingData(raw_data)

#initialize regression coefficients
regression_coefficients = np.zeros((FEATURES,1),dtype=float)

#iterate over the features and compute the regression coefficient
for feature_id in range(FEATURES):

    #identify the feature columns
    X = np.reshape(features[:,feature_id],(RECORDS,1))

    #identify the label columns
    Y = np.reshape(labels[:,feature_id],(RECORDS,1))

    #perform regression on features and labels
    rc,residual,rank,s = np.linalg.lstsq(X,Y)

    #update regression coefficients
    regression_coefficients[feature_id][0] = rc[0][0]

#predictions for the test features
predictions = np.multiply(test_features,regression_coefficients,)

#print predictions
print(predictions)

#save the global predictions to a file locally
processRegressionOutput(regression_coefficients)