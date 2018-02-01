import numpy as np
import matplotlib.pyplot as plt
import pprint

def missingIsNan(s):
	return np.nan if s == b'?' else float(s)

def makeStandardize(X):
	means = X.mean(axis = 0)
	stds = X.std(axis = 0)

	def standardize(origX):
		return (origX - means) / stds

	def unstandardize(stdX):
		return stds * stdX + means

	return (standardize, unstandardize)

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
	# X1 = np.insert(X, 0, 1, 1)

	# 4. Split the data into training (80 %) and testing data (20 %)
	nRows = X.shape[0]
	nTrain = int(round(0.8*nRows)) 
	nTest = nRows - nTrain

	# Shuffle row numbers
	rows = np.arange(nRows)
	np.random.shuffle(rows)

	trainIndices = rows[:nTrain]
	testIndices = rows[nTrain:]

	# Check that training and testing sets are disjoint
	# print(np.intersect1d(trainIndices, testIndices))

	Xtrain = X[trainIndices, :]
	Ttrain = T[trainIndices, :]
	Xtest = X[testIndices, :]
	Ttest = T[testIndices, :]

	# 5. Standardize
	(standardize, unstandardize) = makeStandardize(Xtrain)
	XtrainS = standardize(Xtrain)
	XtestS = standardize(Xtest)

	# 6. Tack column of 1s
	XtrainS1 = np.insert(XtrainS, 0, 1, 1)
	XtestS1 = np.insert(XtestS, 0, 1, 1)

	# 7. Find weights (solve for w) 
	w = np.linalg.lstsq(XtrainS1.T @ XtrainS1, XtrainS1.T @ Ttrain, rcond = None)[0]

	# 8. Predict
	predict = XtestS1 @ w

	# 9. Compute RSME
	rsme = np.sqrt(np.mean((predict - Ttest)**2))

	print(rsme)
