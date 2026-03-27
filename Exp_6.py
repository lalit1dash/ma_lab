#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 


# In[37]:


# Input data
x = np.array([1,2,3,4,5]).reshape(-1, 1)
y = np.array([2, 4, 6, 4, 5]) 


# In[9]:


# Create Linear Regression model
model = LinearRegression()
model.fit(x, y)


# In[10]:


# Get slope and intercept
slope = model.coef_[0]
intercept = model.intercept_
print("Slope:", slope)
print("Intercept:", intercept)


# In[11]:


# Predict values  
y_pred = model.predict(x)


# In[38]:


# Plot the graph
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

