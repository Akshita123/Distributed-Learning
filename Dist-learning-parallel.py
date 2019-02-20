
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
import math
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, wait
executor = ThreadPoolExecutor(max_workers=5)


# In[2]:


def normalEqn(X, y, _lambda, num_core):
    theta = np.array([x[0] for x in np.zeros((8, 1))])
    m = len(y) - 1
    J_min = np.zeros((int(m / num_core), 1))
    for t in range(1, int(m / num_core) + 1):
        theta = pd.DataFrame(np.linalg.inv(
            (
                    (X.iloc[0:num_core * t, :].transpose().dot(X.iloc[0:num_core * t, :]))
                    +
                    (_lambda * np.identity(8))
            )
        )).dot(X.iloc[0:num_core * t, :].transpose().dot(y[0:num_core * t]))
        J_min[t - 1] = ((sum(np.square((X.iloc[0:num_core * t, :].dot(theta)) - y[0:num_core * t]))) / 2) + (
                (_lambda / (2 * m)) * (sum(np.square(theta[0: len(theta) - 1]))))
    return (J_min, theta)


# In[3]:


def computeCost(X, y, theta, theta_avg, i, _lambda, grad, num_core):
    m = len(y) - 1
    J = 0

    J = (np.square((X.iloc[i, :].dot(theta)) - y[i]) / 2)
    + ((_lambda / 2 * m) * sum(np.square(theta[0: len(theta) - 1])))

    grad = ((X.iloc[i, :].dot(theta_avg)) - y[i]) * (X.iloc[i, :])

    grad[0: len(theta) - 1] = grad[0: len(theta) - 1] + ((_lambda / m) * theta_avg[0: len(theta) - 1])

    return (J, grad)


# In[4]:


# Our proposed algorithm
def gradientDescent(X, y, theta, _lambda, num_core):
    m = len(y) - 1
    theta1 = theta
    theta2 = theta
    theta3 = theta
    theta_avg1 = theta
    theta_avg2 = theta
    theta_avg3 = theta
    theta_avg = theta
    J_history = np.zeros((int(m / num_core), 1))
    J_hist = np.zeros((num_core, 1))
    grad = np.array([x[0] for x in np.zeros((8, 1))])
    grad_par1 = grad
    grad_par2 = grad
    grad_par3 = grad
    grad_par4 = grad
    grad_rec1 = grad
    grad_rec2 = grad
    grad_rec3 = grad
    grad_rec4 = grad

    alpha = 1/m
    _eta = 1000
    (J_hist[0], grad_par1) = computeCost(X, y, theta, theta_avg1, 0, _lambda, grad, num_core)
    theta1 = theta1 - (((alpha * _eta * grad_par1) - (alpha * sum(sum(grad_par2, grad_par3), grad_par4))) / (
                _eta + num_core - 1))
    (J_hist[1], grad_par2) = computeCost(X, y, theta, theta_avg2, 1, _lambda, grad, num_core)
    theta2 = theta2 - (((alpha * _eta * grad_par2) - (alpha * sum(sum(grad_par1, grad_par3), grad_par4))) / (
                _eta + num_core - 1))
    (J_hist[2], grad_par3) = computeCost(X, y, theta, theta_avg3, 2, _lambda, grad, num_core)
    theta3 = theta3 - (((alpha * _eta * grad_par3) - (alpha * sum(sum(grad_par2, grad_par1), grad_par4))) / (
                _eta + num_core - 1))
    (J_hist[3], grad_par4) = computeCost(X, y, theta, theta_avg, 3, _lambda, grad, num_core)
    theta = theta - (((alpha * _eta * grad_par4) - (alpha * sum(sum(grad_par2, grad_par3), grad_par1))) / (
                _eta + num_core - 1))

    J_history[0] = sum(J_hist)/num_core
    theta_avg1 = ((_eta * theta1) + sum(sum(theta, theta2), theta3)) / (_eta + num_core - 1)
    theta_avg2 = ((_eta * theta2) + sum(sum(theta1, theta), theta3)) / (_eta + num_core - 1)
    theta_avg3 = ((_eta * theta3) + sum(sum(theta1, theta2), theta)) / (_eta + num_core - 1)
    theta_avg = ((_eta * theta) + sum(sum(theta1, theta2), theta3)) / (_eta + num_core - 1)

    for x in range(num_core, m, num_core):
        #BELOW FOUR STEPS ARE PARALLEL
        data_1 = executor.submit(computeCost, X, y, theta1, theta_avg1, x, _lambda, grad, num_core)
        data_2 = executor.submit(computeCost, X, y, theta2, theta_avg2, x + 1, _lambda, grad, num_core)
        data_3 = executor.submit(computeCost, X, y, theta3, theta_avg3, x + 2, _lambda, grad, num_core)
        data_4 = executor.submit(computeCost, X, y, theta, theta_avg, x + 3, _lambda, grad, num_core)
        done, not_done = wait([data_1, data_2, data_3, data_4])
        done = list(done)
        (J_hist[0], grad_rec1) = done[0].result()
        (J_hist[1], grad_rec2) = done[1].result()
        (J_hist[2], grad_rec3) = done[2].result()
        (J_hist[3], grad_rec4) = done[3].result()
        theta1 = theta1 - (((alpha * _eta * grad_rec1) - (alpha * sum(sum(grad_par2, grad_par3), grad_par4))) / (
                    _eta + num_core - 1))
        theta2 = theta2 - (((alpha * _eta * grad_rec2) - (alpha * sum(sum(grad_par1, grad_par3), grad_par4))) / (
                    _eta + num_core - 1))
        theta3 = theta3 - (((alpha * _eta * grad_rec3) - (alpha * sum(sum(grad_par2, grad_par1), grad_par4))) / (
                    _eta + num_core - 1))
        theta = theta - (((alpha * _eta * grad_rec4) - (alpha * sum(sum(grad_par2, grad_par3), grad_par1))) / (
                    _eta + num_core - 1))

        theta_avg1 = ((_eta * theta1) + sum(sum(theta, theta2), theta3)) / (_eta + num_core - 1)
        theta_avg2 = ((_eta * theta2) + sum(sum(theta1, theta), theta3)) / (_eta + num_core - 1)
        theta_avg3 = ((_eta * theta3) + sum(sum(theta1, theta2), theta)) / (_eta + num_core - 1)
        theta_avg = ((_eta * theta) + sum(sum(theta1, theta2), theta3)) / (_eta + num_core - 1)
        J_history[int(x / 4)] = J_history[int(x / 4) - 1] + (sum(J_hist)/num_core)
    return (J_history, theta)


# In[5]:


data_df = pd.read_csv('test_data.csv', header=None)
X = data_df.iloc[:, :7]
y = data_df.iloc[:, 7]
num_core = 4
m = len(y)
print("Normalizing Features ...\n")

temp_values = X.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(temp_values)
X = pd.DataFrame(x_scaled)

# add a new columns of one
X[len(X.columns)] = 1

print(X.head())

theta = np.array([x[0] for x in np.zeros((8, 1))])

_lambda = 0.001


# In[6]:


(J_history, OGD_theta) = gradientDescent(X, y, theta, _lambda, num_core)
print("Theta found by gradient descent", OGD_theta)


# In[7]:


(J_min, normal_theta) = normalEqn(X, y, _lambda, num_core)
print("Theta found by normal equation", normal_theta);
regret = J_history - J_min
plt.plot(regret)
plt.show()

