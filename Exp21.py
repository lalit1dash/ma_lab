#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Develope a program to implement the naive bayesian classifier considering Olivetti face data set for training compute the accuracy of the classifier considering a few test data sets. 


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


# In[3]:


data = fetch_olivetti_faces()

X = data.data        # 4096 features (64x64 image flattened)
y = data.target      # 40 classes (persons)


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# In[5]:


model = GaussianNB()
model.fit(X_train, y_train)


# In[6]:


y_pred = model.predict(X_test)


# In[7]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[8]:


print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[9]:


# Display some test images with predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(64, 64), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}")
    ax.axis('off')

plt.show()


# In[ ]:




