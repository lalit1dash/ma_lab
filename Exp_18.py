#!/usr/bin/env python
# coding: utf-8

# In[1]:


# wap to implement the naive bayesian classifier for a sample training dataset stored as a .csv file compute the accuracy of the classifier considering few test datasets. 


# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[2]:


# Load dataset
data = pd.read_csv("iris.csv")

# Show first 5 rows
print(data.head())


# In[3]:


# Features (independent variables)
X = data.iloc[:, :-1].values   # all columns except last

# Target (dependent variable)
y = data.iloc[:, -1].values    # last column

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # 30% test data
    random_state=42     # for reproducibility
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# In[5]:


# Create model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)



# In[6]:


# Predict on test data
y_pred = model.predict(X_test)

print("Predictions:", y_pred[:10])


# In[9]:


# Example test samples
sample_data = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.0, 5.2, 2.3],
    [5.9, 3.2, 4.8, 1.8]
])

sample_pred = model.predict(sample_data)

print("Custom Sample Predictions:", sample_pred)


# In[ ]:




