#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Define the function y = (x + 3)^2
def func(x):
    return (x + 3)**2


# In[3]:


# Define the derivative of the function
def gradient(x):
    return 2 * (x + 3)


# In[4]:


def gradient_descent(starting_point, learning_rate, num_iterations):
    x_values = []
    y_values = []
    x = starting_point

    for _ in range(num_iterations):
        x_values.append(x)
        y_values.append(func(x))
        x = x - learning_rate * gradient(x)

    return x_values, y_values


# In[5]:


starting_point = 2
learning_rate = 0.1
num_iterations = 50


# In[14]:


# Run Gradient Descent
x_values, y_values = gradient_descent(starting_point, learning_rate, num_iterations)


# In[15]:


# Plot the function and the path taken by Gradient Descent
x_range = np.linspace(-8, 2, 100)
plt.plot(x_range, func(x_range), label='Function: $y=(x+3)^2$')
plt.scatter(x_values, y_values, color='red', label='Gradient Descent Path')
plt.title('Gradient Descent to Find Local Minima')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[ ]:




