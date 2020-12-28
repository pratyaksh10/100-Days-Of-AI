# DAY 1: SIMPLE LINEAR REGRESSION 

Linear regression is a linear model that computes output variable ‘y’ using a linear combination of input variable ‘x’. For instance, a simple linear regression model is shown –
<p align="center">
  <img src="https://github.com/pratyaksh10/100-Days-Of-AI/blob/master/Day1/eq.png">
</p>


## HOW TO MAKE PREDICTIONS USING LINEAR REGRESSION?
In this model, we are trying to minimize the error by computing the best fit. Our goal is to optimize the weights by minimizing the length between observed output (Yo) and predicted output (Yp).

Firstly, we calculate the mean and the variance of both the input and output variables from the training data. 
```python
def mean(n):     #Mean
    return np.sum(n)/float(len(n))

def varience(n, mean):    #Varience
    return sum([np.square((i - mean)) for i in n])
```

Secondly, we estimate the covariance of the two groups of numbers. The covariance describes how those numbers change together.
```python
def covarience(x,y,x_mean,y_mean):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - x_mean)*(y[i] - y_mean)
    return covar
```
Lastly, we estimate the coefficients b0 and b1 of our linear regression model.
```python
numerator = covarience(X,Y, x_mean,y_mean)
denominator = varience(X, x_mean)

b_1 = numerator/denominator
b_0 = y_mean - (b_1*x_mean)
print(b_1,b_0)
```

## DATASET 

### Pizza Franchise
In the following data

X = annual franchise fee ($1000)

Y = start up cost ($1000) for a pizza franchise

Reference: Business Opportunity Handbook

Visit [https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html](https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html) to download the dataset.

## FILES 
- [`LinearRegression.py`](LinearRegression.py) : Contains a python file that contains the implemention of a simple Linear Regression model without using Tensorflow, Keras, etc.
- [`dataset.csv`](dataset.csv) : Contains the dataset used to train the model.  

## VISUALIZING THE TEST RESULTS
The final step is to visualize the test results and evaluate our model. We use matplotlib.pyplot to make Scatter Plots of our dataset and the Liner Regression plot shows how close is our model prediction. The following figure illustrates the scatter plot and prediction results.
<p align="center">
  <img src="https://github.com/pratyaksh10/100-Days-Of-AI/blob/master/Day1/eq.png">
</p>
