#!/usr/bin/env python
# coding: utf-8

# Implement Dimensionality reduction using principle component Analysis(PCA) method on a dataset. 

# In[14]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.datasets import load_iris


# In[34]:


#Load dataset
data = load_iris()
X = data.data
y=data.target
t=data.target_names
print("Original Shape:",X.shape)


# In[18]:


#Standardized data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[19]:


#
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled) 


# In[21]:


explained_varience=pca.explained_variance_ratio_ 
print(f"Total explained variance by 2 components:{np.sum(explained_varience)}") 


# In[33]:


plt.figure(figsize=(6,5))
for target,colour in zip(range(len(t)),["orange","blue","green"]):
    plt.scatter(X_pca[y==target,0],X_pca[y==target,1],c=colour,label=t[target],s=20)


# In[ ]:




