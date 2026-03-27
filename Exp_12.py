#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


import seaborn as sns
from sklearn.datasets import make_blobs 


# In[3]:


x,y=make_blobs(
    n_samples=1000,
    n_features=2, 
    centers=4,
    random_state=42
)


# In[4]:


sns.scatterplot(x=x[:,0],y=x[:,1])


# In[5]:


from sklearn.cluster import KMeans 
k=4
kmeans=KMeans(
    n_clusters=k,
    random_state=42
)


# In[7]:


labels=kmeans.fit_predict(x)
labels


# In[8]:


sns.scatterplot(x=x[:,0],y=x[:,1],c=labels)


# In[9]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate synthetic 2D data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 2. Define K values to test
k_values = [1, 3, 5]

# 3. Plotting the results
plt.figure(figsize=(15, 5))

for i, k in enumerate(k_values):
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)

    # Create subplot
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    # Plot cluster centers
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title(f'K-Means Clustering (K={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
# Save or show the plot
plt.show()


# In[ ]:





# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([
    [1,2],[1.5,1.8],[5,8],[8,8],
    [1,0.6],[9,11],[8,2],[10,2],[9,3]
])

# K-Means function
def kmeans(X, k, max_iter=100):

    # Random centroid initialization
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):

        # Assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Compute new centroids
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

        # Stop if centroids stop changing
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Calculate SSE
    sse = 0
    for i in range(len(X)):
        sse += np.linalg.norm(X[i] - centroids[clusters[i]])**2

    return clusters, centroids, sse


# Different K values
k_values = [1,3,5]
results = {}

plt.figure(figsize=(15,4))

for i,k in enumerate(k_values):

    clusters, centroids, sse = kmeans(X,k)
    results[k] = sse

    plt.subplot(1,3,i+1)
    plt.scatter(X[:,0], X[:,1], c=clusters, cmap='rainbow')
    plt.scatter(centroids[:,0], centroids[:,1], c='black', marker='X', s=200)
    plt.title(f"K = {k}")

plt.show()

# Print comparison
print("Comparison using SSE (Lower is better):")
for k in results:
    print("K =",k," -> SSE =",results[k])


# In[ ]:




