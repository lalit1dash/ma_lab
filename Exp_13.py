#!/usr/bin/env python
# coding: utf-8

# In[1]:


# implement of simple and multiple linear Regression 


# # Simple linear Regression 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[3]:


np.random.seed(42)

X = np.random.rand(50, 1) * 100  

Y = 3.5 * X + np.random.randn(50, 1) * 20


# In[4]:


model = LinearRegression()
model.fit(X, Y)


# In[5]:


Y_pred=model.predict(X)


# In[6]:


plt.figure(figsize=(8,6)) 
plt.scatter(X, Y, color='blue', label='Data Points') 
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line') 
plt.title('Linear Regression on Random Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()


# In[7]:


print("Slope (Coefficient):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])


# # Multiple Linear Regression 

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing


# In[9]:


california_housing = fetch_california_housing()

X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = pd.Series(california_housing.target)


# In[10]:


X = X[['MedInc', 'AveRooms']]


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[12]:


model=LinearRegression() 
model.fit(X_train,y_train)


# In[13]:


y_pred=model.predict(X_test)


# In[14]:


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test['MedInc'], X_test['AveRooms'],
           y_test, color='blue', label='Actual Data')

x1_range = np.linspace(X_test['MedInc'].min(), X_test['MedInc'].max(), 100)
x2_range = np.linspace(X_test['AveRooms'].min(), X_test['AveRooms'].max(), 100)
x1, x2 = np.meshgrid(x1_range, x2_range)

z = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)

ax.plot_surface(x1, x2, z, color='red', alpha=0.5, rstride=100, cstride=100)

ax.set_xlabel('Median Income')
ax.set_ylabel('Average Rooms')
ax.set_zlabel('House Price')
ax.set_title('Multiple Linear Regression Best Fit Line (3D)')

plt.show()


# In[ ]:





# In[15]:


import numpy as np
import matplotlib.pyplot as plt

# Dataset
x = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

# Mean
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate slope (b1)
numerator = np.sum((x - x_mean)*(y - y_mean))
denominator = np.sum((x - x_mean)**2)

b1 = numerator / denominator

# Calculate intercept (b0)
b0 = y_mean - b1*x_mean

print("Slope:", b1)
print("Intercept:", b0)

# Prediction
y_pred = b0 + b1*x

# Plot
plt.scatter(x,y)
plt.plot(x,y_pred,color="red")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression")
plt.show()


# In[16]:


import numpy as np

# Dataset
X = np.array([
    [1,1,2],
    [1,2,3],
    [1,3,4],
    [1,4,5]
], dtype=float)   # ensure float type

Y = np.array([6,8,10,12], dtype=float)

# Normal Equation using pseudo-inverse
beta = np.linalg.pinv(X.T @ X) @ X.T @ Y

print("Regression Coefficients:")
print(beta)

# Prediction
new_data = np.array([1,5,6], dtype=float)
prediction = new_data @ beta

print("Predicted value:", prediction)


# In[ ]:




