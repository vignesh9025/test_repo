import numpy as np
import matplotlib.pyplot as plt
import pprint as pp

def missingIsNan(s):
	return np.nan if s == b'?' else float(s)

if __name__ == '__main__':
	# 1. Load the data.
	data = np.loadtxt("Data\\auto-mpg.data", usecols = range(8), converters = {3: missingIsNan})

	# 2. 'Clean' the data.
	Cdata = data[~np.isnan(data).any(axis = 1)]

	# 3. Split it into input (X) and target (T)
	# 	 Target = mpg (first column)
	#	 Input = remaining - columns 2 to 7
	T = Cdata[:, 0:1]
	X = Cdata[:, 1:]
	
	# 4. Append column of 1s to X
	X1 = np.insert(X, 0, 1, 1)

	# 5. Split the data into training (80 %) and testing data (20 %)
	nRows = X1.shape[0]
	nTrain = int(round(0.8*nRows)) 
	nTest = nRows - nTrain

	# Shuffle row numbers
	rows = np.arange(nRows)
	np.random.shuffle(rows)

	trainIndices = rows[:nTrain]
	testIndices = rows[nTrain:]

	# Check that training and testing sets are disjoint
	# print(np.intersect1d(trainIndices, testIndices))

	Xtrain = X1[trainIndices, :]
	Ttrain = T[trainIndices, :]
	Xtest = X1[testIndices, :]
	Ttest = T[testIndices, :]

	# 6. Find weights 
	w = np.linalg.lstsq(Xtrain.T @ Xtrain, Xtrain.T @ Ttrain, rcond = None)

	# Keep only weights and discard other information.other
	w = w[0]

	# 7. Use model (weight) to predict (test).
	predict = Xtest @ w

	# 8. Compare to see how well we've done (plot) and RMSE.
	rmse = np.sqrt(np.mean((predict - Ttest)**2))

	print(rmse)
