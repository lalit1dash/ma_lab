#!/usr/bin/env python
# coding: utf-8

# # wap to implement simple linear regression for iris using sklearn and plot confusion matrix

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[5]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report


# In[6]:


iris = load_iris()
X = iris.data      
y = iris.target   


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# In[8]:


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)


# In[10]:


print("Accuracy:", accuracy_score(y_test, y_pred))


# In[11]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# In[12]:


plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    fmt="d"
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Iris Dataset") 
plt.show()


# In[13]:


print(classification_report(y_test,y_pred))

