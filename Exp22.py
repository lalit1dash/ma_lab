#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Develope a program to implement k-means clustering using wisconsin breast canser data set and visualize the clustering result.


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[14]:


data = load_breast_cancer()

X = data.data      # Features
y = data.target    # Actual labels (not used in clustering, only for comparison)


# In[20]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[21]:


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

clusters = kmeans.labels_


# In[22]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# In[23]:


plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.title("K-Means Clustering (Breast Cancer Dataset)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# In[24]:


plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title("Actual Labels (Ground Truth)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




