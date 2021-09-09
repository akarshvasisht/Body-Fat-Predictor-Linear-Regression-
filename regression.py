import numpy as np
from numpy.linalg import inv
import csv
from matplotlib import pyplot as plt
import random
import math


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):  
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        dataset = []       
        for row in reader:
            rowDetails = []
            row.pop('IDNO')
            for i in row:
                rowDetails.append(float(row[i]))  
                
            dataset.append(rowDetails)
    
    return np.array(dataset)
    

def print_stats(dataset, col):
    num_data = len(dataset)
    colVals = []
    
    for i in dataset:
        colVals.append(i[col])

    print(len(dataset))
    print(round(np.mean(colVals), 2))
    print(round(np.std(colVals), 2))
    
    return

def regression(dataset, cols, betas):
    mse = 0
    
    for row in dataset:
        rowSum = 0
        
        for i in cols: 
            xi = row[i] * betas[cols.index(i) + 1]
            rowSum += xi
            
        mse += (betas[0] + rowSum - row[0]) ** 2

    mse = mse / float(len(dataset))
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    der_list = []
    
    for beta in betas:
        der_mse = 0
        
        for row in dataset:
            rowSum = 0
            
            for i in cols: 
                xi = row[i] * betas[cols.index(i) + 1]
                rowSum += xi
                
                if len(der_list) == 0:
                    tempvar = 1.0
                elif cols.index(i) + 1 == len(der_list):
                    tempvar = row[i]
            der_mse += (betas[0] + rowSum - row[0]) * tempvar
    
        der_mse = (der_mse / float(len(dataset))) * 2
        der_list.append(der_mse)
        
    return np.array(der_list)


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    curMSE = gradient_descent(dataset, cols, betas)
    curBetas = betas
    
    for t in range(1, T + 1):
        finalString = ""
        for i in range(len(curBetas)):
            curBetas[i] = curBetas[i] - eta * curMSE[i]
        
        finalString += str(t) + " " + str(round(regression(dataset, cols, betas), 2)) +  " "
        for j in range(len(curBetas)):
            finalString += str(round(curBetas[j], 2)) + " "
        
        print(finalString)
        curMSE = gradient_descent(dataset, cols, curBetas)
        
    return

def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    yVals = []
    for row in dataset:
        yVals.append(row[0])
        
    XVals = []
    for i in range(len(dataset)):
        XVals.append([1])
        for j in range(len(cols)):
            XVals[i].append(dataset[i][cols[j]])
                
    XVals = np.array(XVals)
    yVals = np.array(yVals)
    X_transpose = np.transpose(XVals)
    X_transpose_X = np.dot(X_transpose, XVals)
    XTX_inverse = inv(X_transpose_X)
    result = np.dot(np.dot(XTX_inverse, X_transpose), yVals)
    
    mse = regression(dataset, cols, result)
    
    final = []
    final.append(mse)
    for element in result:
        final.append(element)
    final = tuple(final)
    
        
    return final   


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)
    
    result = 0
    for i in range(len(features)):
        result += betas[i + 2] * features[i]
        
    result += betas[1]
        
    return result

def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
        
    
    linear = []
    for i in range(len(X)):
        linear_list = []
        linear_list.append(float(X[i][0]))
        linear.append(linear_list)
        
    mu = 0   
    for j in range(len(linear)):
        yi = betas[0] + betas[1] * linear[j][0] + np.random.normal(mu, sigma)
        linear[j].insert(0, yi)
        
    quadratic = []
    for i in range(len(X)):
        quadratic_list = []
        quadratic_list.append(float(X[i][0]))
        quadratic.append(quadratic_list)
        
    for j in range(len(quadratic)):
        yi = alphas[0] + alphas[1] * math.pow(quadratic[j][0], 2) + np.random.normal(mu, sigma)
        quadratic[j].insert(0, yi)
        
    return (np.array(linear), np.array(quadratic))

def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    input_arr = []
    for i in range(1000):
        XList = []
        XList.append(random.uniform(-100.0, 100.0))
        input_arr.append(XList)
    
    betas = [1, 2]
    alphas = [3, 4]
    
    sigList = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    
    linearMSE = []
    quadraticMSE = []
    
    for sigma in sigList:
        dsTuple = synthetic_datasets(np.array(betas), np.array(alphas), np.array(input_arr), sigma)
        
        linearBetas = compute_betas(dsTuple[0], cols=[1])
        quadraticBetas = compute_betas(dsTuple[1], cols=[1])
        
        linearBetas = linearBetas[1:]
        quadraticBetas = quadraticBetas[1:]
        
        lmse = regression(dsTuple[0], cols=[1], betas=linearBetas)
        qmse = regression(dsTuple[0], cols=[1], betas=quadraticBetas)
        
        linearMSE.append(lmse)
        quadraticMSE.append(qmse)
        
    plt.plot(sigList, linearMSE, marker='o',label = "Linear MSE")
    plt.yscale('log',base= 10) 
    plt.xscale('log',base= 10) 
  

    plt.plot(sigList, quadraticMSE, marker='o', label= "Quadratic MSE")
    plt.yscale('log',base=10) 
    plt.xscale('log',base=10) 
    
    plt.xlabel("Sigma")
    plt.ylabel("MSE")
    
    plt.legend()

    plt.savefig('mse.pdf')
    
    return

if __name__ == '__main__':
    dataset=get_dataset('bodyfat.csv')
    print(dataset.shape)
    print_stats(dataset, 1)
    print(regression(dataset, cols=[2,3,4], betas=[0,-1.1,-.2,3]))
    print(gradient_descent(dataset, cols=[2,3], betas=[0,0,0]))
    print(iterate_gradient(dataset, cols=[1,8], betas=[400,-400,300], T=10, eta=1e-4))
    print(compute_betas(dataset, cols=[1,2]))
    print(synthetic_datasets(np.array([0,2]), np.array([0,1]), np.array([[4]]), 1))
    plot_mse()
    
