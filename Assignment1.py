import numpy as np
import matplotlib.pyplot as plt

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
    Xs = np.insert(Xs, 0, 1, 1)
    ncolsX = Xs.shape[1]
    # Initialize weights to zero.
    w = np.zeros((ncolsX, ncolsT))

    for i in range(numberOfIterations):
        for n in range(nSamples):
            predicted = Xs[n:n+1, :] @ w
            w += learningRate * Xs[n:n+1, :].T * (T[n:n+1, :] - predicted)

    dict = {'means': means, 'stds': std, 'w': w}
    return dict


if __name__ == '__main__':
	X = np.array([[0,1,2],[3,4,5], [5,6,7]])
	T = np.array([[1,2,3]])
	T = np.transpose(T)	

	model = train(X, T)
	predict = use(model, X)
	r = rmse(predict, T)

	model2 = trainSGD(X, T, 0.1, 100)
	print(model2)
