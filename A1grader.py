import os
import numpy as np

print('\n======================= Code Execution =======================\n')

if False:
    runningInNotebook = False
    print('========================RUNNING INSTRUCTOR''S SOLUTION!')
    import A1mysolution as useThisCode
    train = useThisCode.train
    trainSGD = useThisCode.trainSGD
    use = useThisCode.use
    rmse = useThisCode.rmse

else:
    print('Extracting python code from notebook and storing in notebookcode.py')
    import subprocess
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         '*-A1.ipynb', '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.ClassDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *

def close(a, b, within=0.01):
    return abs(a-b) < within

g = 0

A = np.array([[1,2,3], [4,5,6]])
B = A + 1
print('Testing rmse(A, B) with\n A =\n{}\n and B =\n{}'.format(A, B))

try:
    answer = rmse(A, B)
    correctAnswer = 1
    if close(answer, correctAnswer):
        g += 10
        print('\n--- 10/10 points. Correctly returned {}'.format(answer))
    else:
        print('\n---  0/10 points. Incorrect. You returned {}, but correct answer is {}'.format(answer, correctAnswer))
except Exception as ex:
    print('\n--- 0/10 points. rmse raised the exception\n {}'.format(ex))


X = np.arange(15).reshape((5,3))
X[3:5, :] *= 2
T = X[:,0:2] + 0.1 * X[:,1:2] * X[:,2:3]

print('\nTesting model = train(X, T) with\n X=\n{}\n and T=\n{}'.format(X, T))
try:
    model = train(X, T)
    if 'means' in model.keys():
        g += 5
        print('\n--- 5/5 points. Model correctly includes a key named \'means\'.')
    else:
        print('\n--- 0/5 points. Model does not include a key named \'means\'.')
        
    if 'stds' in model.keys():
        g += 5
        print('\n--- 5/5 points. Model correctly includes a key named \'stds\'.')
    else:
        print('\n--- 0/5 points. Model does not include a key named \'stds\'.')
        
    if 'w' in model.keys():
        g += 5
        print('\n--- 5/5 points. Model correctly includes a key named \'w\'.')
    else:
        print('\n--- 0/5 points. Model does not include a key named \'w\'.')

except Exception as ex:
    print('\n--- 0/15 points. train raised the exception\n {}'.format(ex))

print('\nTesting rmse(T, use(model, X))')
try:
    answer = rmse(T, use(model, X))
    correctAnswer = 5.24
    if close(answer, correctAnswer, 0.2):
        g += 15
        print('\n--- 15/15 points. Error is correctly calculated as {}.'.format(answer))
    else:
        print('\n---  0/15 points. Error of {} is wrong.  It should be {}.'.format(answer, correctAnswer))

except Exception as ex:
    print('\n--- 0/15 points. rmse or use raised the exception\n {}'.format(ex))



print('\nTesting model = trainSGD(X, T, 0.01, 1000) with\n X=\n{}\n and T=\n{}'.format(X, T))
try:
    model = trainSGD(X, T, 0.01, 1000)
    if 'means' in model.keys():
        g += 5
        print('\n--- 5/5 points. Model correctly includes a key named \'means\'.')
    else:
        print('\n--- 0/5 points. Model does not include a key named \'means\'.')
        
    if 'stds' in model.keys():
        g += 5
        print('\n--- 5/5 points. Model correctly includes a key named \'stds\'.')
    else:
        print('\n--- 0/5 points. Model does not include a key named \'stds\'.')
        
    if 'w' in model.keys():
        g += 5
        print('\n--- 5/5 points. Model correctly includes a key named \'w\'.')
    else:
        print('\n--- 0/5 points. Model does not include a key named \'w\'.')

except Exception as ex:
    print('\n--- 0/15 points. trainSGD raised the exception\n {}'.format(ex))

print('\nTesting rmse(T, use(model, X))')
try:
    answer = rmse(T, use(model, X))
    correctAnswer = 5.24
    if close(answer, correctAnswer):
        g += 15
        print('\n--- 15/15 points. Error is correctly calculated as {}.'.format(answer))
    else:
        print('\n---  0/15 points. Error of {} is wrong.  It should be {}.'.format(answer, correctAnswer))

except Exception as ex:
    print('\n--- 0/15 points. rmse or use raised the exception\n {}'.format(ex))


name = os.getcwd().split('/')[-1]

print('\n{} Execution Grade is {}/70'.format(name, g))

print('\n======================= Plots and Descriptions =======================')

print('\n--- _/5 points. Descriptions of data, including plots.')

print('\n--- _/5 points. Descriptions of algorithms for fitting linear model.')

print('\n--- _/5 points. Descriptions of code for all defined functions.')

print('\n--- _/5 points. Plots of predictions made by models from train and trainSGD. Must at least include predicted values versus actual values for each target variable and for each model.')

print('\n--- _/5 points. Discussions of the above plots of predictions and actual values.')

print('\n--- _/5 points. Discussion of accuracy of each model.  Refer to RMSE values and what they mean with respect to the range of target values.')



print('\n{} Notebook Grade is __/30'.format(name))

print('\n{} FINAL GRADE is __/100'.format(name))



