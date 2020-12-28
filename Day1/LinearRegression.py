#Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Calculating Mean and Varience

def mean(n):     #Mean

    return np.sum(n)/float(len(n))

def varience(n, mean):    #Varience

    return sum([np.square((i - mean)) for i in n])


def covarience(x,y,x_mean,y_mean):

    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - x_mean)*(y[i] - y_mean)

    return covar


#Loading Dataset

dataset = pd.read_csv('dataset.csv')

X = dataset['X'].values
Y = dataset['Y'].values

x_mean = mean(X)
y_mean = mean(Y)

#Calculating coefficients

numerator = covarience(X,Y, x_mean,y_mean)
denomenator = varience(X, x_mean)

b_1 = numerator/denomenator
b_0 = y_mean - (b_1*x_mean)

print(b_1,b_0)


#plot

x_max = np.max(X) + 100
x_min = np.min(X) - 100

#calculating line values

x = np.linspace(x_min, x_max, 500)
y = b_0 + b_1*x

#plotting line
plt.plot(x, y, color='#00ff00', label='Linear Regression')
#plot the data point
plt.scatter(X, Y, color='#ff0000', label='Data Point')
# x-axis label
plt.xlabel('Annual franchise fee ($1000)')
#y-axis label
plt.ylabel('Start up cost ($1000)')
plt.legend()
plt.show()



