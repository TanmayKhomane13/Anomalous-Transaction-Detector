import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def estimateGaussian(X):
    """
    Calculates mean and variance of all features in the dataset

    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n, ) Mean of all features
        var (ndarray): (n, ) Variance of all features
    """

    m, n = X.shape
    mu = np.mean(X, axis = 0)
    var = np.var(X, axis = 0)

    return mu, var

def findAdaptiveThreshold(pVal):
    """
    Finds an adaptive threshold based on the largest gap in sorted probabilities

    Args:
        pVal(ndarray): Probability values of data points
    Returns:
        epsilon (float): Adaptive threshold value
    """
    sortedpVals = np.sort(pVal)
    differences = np.diff(sortedpVals)
    jumpIndex = np.argmax(differences) # find the index of the biggest jump

    epsilon = sortedpVals[jumpIndex] # use it as the threshold

    return epsilon

df = pd.read_csv('transactions.csv')

mu, var = estimateGaussian(df.values)

pValues = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((df.values - mu) ** 2) / (2 * var))

pValues = pValues.flatten()

epsilon = findAdaptiveThreshold(pValues)

# detect anomalies
df['Anomaly'] = (pValues < epsilon).astype(int)

print("Anomalous Transactions: ")
print(df[df['Anomaly'] == 1])

# plotting
plt.figure(figsize = (10, 5))
plt.scatter(df.index, df['Transaction Amount'], color = 'blue', label = 'Normal Transactions')
plt.scatter(df[df['Anomaly'] == 1].index, df[df['Anomaly'] == 1]['Transaction Amount'], color = 'red', label = 'Anomalies', marker = 'o', edgecolors = 'black', s = 100)
plt.title('Anomalous Transactions Detection')
plt.xlabel('Transaction Index')
plt.ylabel('Transaction Amount')
plt.legend()
plt.grid(True)
plt.show()
