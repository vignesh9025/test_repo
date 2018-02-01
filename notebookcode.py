
# coding: utf-8

# $\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}$

# # Assignment 1: Linear Regression

# *by Vignesh M. Pagadala*

# ## Overview

# Describe the objective of this assignment, and very briefly how you accomplish it.  Say things like "linear model", "samples of inputs and known desired outputs" and "minimize the sum of squared errors". DELETE THIS TEXT AND INSERT YOUR OWN.

# ## Method

# Define in code cells the following functions as discussed in class.  Your functions' arguments and return types must be as shown here.
# 
#   * ```model = train(X, T)```
#   * ```predict = use(model, X)```
#   * ```error = rmse(predict, T)```
#   
# Let ```X``` be a two-dimensional matrix (```np.array```) with each row containing one data sample, and ```T``` be a two-dimensional matrix of one column containing the target values for each sample in ```X```.  So, ```X.shape[0]``` is equal to ```T.shape[0]```.   
# 
# Function ```train``` must standardize the input data in ```X``` and return a dictionary with  keys named ```means```, ```stds```, and ```w```.  
# 
# Function ```use``` must also standardize its input data X by using the means and standard deviations in the dictionary returned by ```train```.
# 
# Function ```rmse``` returns the square root of the mean of the squared error between ```predict``` and ```T```.
# 
# Also implement the function
# 
#    * ```model = trainSGD(X, T, learningRate, numberOfIterations)```
# 
# which performs the incremental training process described in class as stochastic gradient descent (SGC).  The result of this function is a dictionary with the same keys as the dictionary returned by the above ```train``` function.

# In[16]:


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


# In this section, ilatex math formulas defining the formula that is being minimized, and the matrix calculation for finding the weights. 
# 
# In this section, include all necessary imports and the function definitions. Also include some math formulas using latex syntax that define the formula being minimized and the calculation of the weights using a matrix equation.  You do not need to include the math formulas showing the derivations.

# ## Examples

# In[13]:


# from A1mysolution import *


# In[18]:


import numpy as np

X = np.arange(10).reshape((5,2))
T = X[:,0:1] + 2 * X[:,1:2] + np.random.uniform(-1, 1,(5, 1))
print('Inputs')
print(X)
print('Targets')
print(T)


# In[19]:


model = train(X, T)
model


# In[20]:


predicted = use(model, X)
predicted


# In[21]:


rmse(predicted, T)


# In[22]:


modelSGD = trainSGD(X, T, 0.01, 100)
modelSGD


# In[23]:


predicted = use(modelSGD, X)
predicted


# In[24]:


rmse(predicted, T)


# ## Data

# Download ```energydata_complete.csv``` from the [Appliances energy prediction Data Set ](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) at the UCI Machine Learning Repository. Ignore the first column (date and time), use the next two columns as target variables, and use all but the last two columns (named rv1 and rv2) as input variables. 
# 
# In this section include a summary of this data, including the number of samples, the number and kinds of input variables, and the number and kinds of target variables.  Also mention who recorded the data and how.  Some of this information can be found in the paper that is linked to at the UCI site for this data set.  Also show some plots of target variables versus some of the input variables to investigate whether or not linear relationships might exist.  Discuss your observations of these plots.

# ## Results

# Apply your functions to the data.  Compare the error you get as a result of both training functions.  Experiment with different learning rates for ```trainSGD``` and discuss the errors.
# 
# Make some plots of the predicted energy uses and the actual energy uses versus the sample index.  Also plot predicted energy use versus actual energy use.  Show the above plots for the appliances energy use and repeat them for the lights energy use. Discuss your observations of each graph.
# 
# Show the values of the resulting weights and discuss which ones might be least relevant for fitting your linear model.  Remove them, fit the linear model again, plot the results, and discuss what you see.

# ## Grading
# 
# Your notebook will be run and graded automatically.  Test this grading process by first downloading [A1grader.tar](http://www.cs.colostate.edu/~anderson/cs445/notebooks/A1grader.tar) and extract `A1grader.py` from it. Run the code in the following cell (after deleting the one containing A1mysolution) to demonstrate an example grading session.  You should see a perfect execution score of 70/70 if your functions are defined correctly. The remaining 30 points will be based on the results you obtain from the energy data and on your discussions.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook.  It will include additional tests.  You need not include code to test that the values passed in to your functions are the correct form.  

# In[25]:


get_ipython().run_line_magic('run', '-i "A1grader.py"')


# ## Check-in

# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A1.ipynb```.  So, for me it would be ```Anderson-A1.ipynb```.  Submit the file using the ```Assignment 1``` link on [Canvas](https://colostate.instructure.com/courses/41327).
# 
# Grading will be based on 
# 
#   * correct behavior of the required functions listed above,
#   * easy to understand plots in your notebook,
#   * readability of the notebook,
#   * effort in making interesting observations, and in formatting your notebook.

# ## Extra Credit

# Download a second data set and repeat all of the steps of this assignment on that data set.
