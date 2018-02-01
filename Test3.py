import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint as pp

def train(X, T):
    # Standardize input data (X)

    # Calculate mean and std.
    means = X.mean(axis = 0)
    std = X.std(axis = 0)

    Xs = (X - means) / std

    # Tack a column of 1s
    Xs = np.insert(Xs, 0, 1, 1)

    # Use Xs to generate model (w)
    w = np.linalg.lstsq(Xs.T @ Xs, Xs.T @ T, rcond = None)[0]

    # Return as a dictionary
    dict = {'means': means, 'stds': std, 'w': w}
    return dict

def use(model, X):
    # Use model and input X to predict.
    means = model['means']
    std = model['stds']
    w = model['w']

    # Standardize X
    Xs = (X - means) / std

    # Tack column of 1s
    Xs = np.insert(Xs, 0, 1, 1)

    # Predict
    predict = Xs @ w
    #print(Xs.shape)
    #print(w.shape)
    return predict

def rmse(predict, T):
    rmerr = np.sqrt(np.mean((T - predict) ** 2))
    return rmerr

def trainSGD(X, T, learningRate, numberOfIterations):
    # Standardize inputs X.
    means = X.mean(axis = 0)
    std = X.std(axis = 0)
    Xs = (X - means) / std

    nSamples = Xs.shape[0]
    ncolsT = T.shape[1]

    # Tack a column of 1s
    Xs1 = np.insert(Xs, 0, 1, 1)
    ncolsX = Xs1.shape[1]
    # Initialize weights to zero.
    w = np.zeros((ncolsX, ncolsT))

    for i in range(numberOfIterations):
        for n in range(nSamples):
            predicted = Xs1[n:n+1, :] @ w
            w += learningRate * Xs1[n:n+1, :].T * (T[n:n+1, :] - predicted)
            #print(w)	
    dict = {'means': means, 'stds': std, 'w': w}
    return dict

# Load the csv data.
dframe=pd.read_csv('Data\\energydata_complete.csv', sep=',',header=None)
# Filter out required columns.
dframe = dframe.drop(dframe.columns[[0, -2, -1]], axis=1)
# Input data columns
Xlabels = []
Xl = dframe.iloc[0, 2:]
for i in Xl:
	Xlabels.append(i)
# Target data columns
Tlabels = []
Tl = dframe.iloc[0, :2]
for i in Tl:
	Tlabels.append(i)

# Get target.
Td = dframe.iloc[1:, [0,1]]
Td = Td.as_matrix()
Tenergy = Td.astype(float)

# Get input.
Xd = dframe.iloc[1:, 2:]
Xd = Xd.as_matrix()
Xenergy = Xd.astype(float)

# Split into training (80 %) and testing data (20 %).
nRows = Xenergy.shape[0]
nTrain = int(round(0.8*nRows)) 
nTest = nRows - nTrain

# Shuffle row numbers
rows = np.arange(nRows)
np.random.shuffle(rows)

trainIndices = rows[:nTrain]
testIndices = rows[nTrain:]

Xtrain = Xenergy[trainIndices, :]
Ttrain = Tenergy[trainIndices, :]
Xtest = Xenergy[testIndices, :]
Ttest = Tenergy[testIndices, :]
'''
# Use functions to predict.
# 1. Using first function.
# Train
model = train(Xtrain, Ttrain)
# Use on test data.
predict = use(model, Xtest)
# Calculate RMSE
rmerr = rmse(predict, Ttest)

print("The RMSE with first function: ")
print(rmerr)
'''
# 2. Using the second function.
model2 = trainSGD(Xtrain, Ttrain, 0.01, 10)
predict2 = use(model2, Xtest)
rmerr2 = rmse(predict2, Ttest)

#print(model2)
#print("The RMSE with the second function: ")
print(predict2[:,0])

