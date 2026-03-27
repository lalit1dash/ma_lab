#!/usr/bin/env python
# coding: utf-8

# Q5 wap to demonstrate the working of the decesion tree based ID3 algorithm by considering a dataset.

# In[1]:


import pandas as pd
import numpy as np
from math import log2


# In[2]:


df=pd.read_csv("weather.csv") 
df


# In[3]:


# Function to calculate entropy
def entropy(target_col):
    values, counts = np.unique(target_col, return_counts=True)
    entropy_val = 0
    for count in counts:
        probability = count / sum(counts)
        entropy_val -= probability * log2(probability)
    return entropy_val


# In[4]:


# Function to calculate information gain
def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    values, counts = np.unique(df[feature], return_counts=True)

    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset = df[df[feature] == value]
        weighted_entropy += (count / sum(counts)) * entropy(subset[target])

    return total_entropy - weighted_entropy


# In[5]:


# ID3 algorithm
def id3(df, target, features):
    if len(np.unique(df[target])) == 1:
        return np.unique(df[target])[0]

    if len(features) == 0:
        return df[target].mode()[0]

    gains = [information_gain(df, feature, target) for feature in features]
    best_feature = features[np.argmax(gains)]

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in np.unique(df[best_feature]):
        subset = df[df[best_feature] == value]
        tree[best_feature][value] = id3(subset, target, remaining_features)

    return tree


# In[7]:


# Build decision tree
features = list(df.columns[:-1])
decision_tree = id3(df, 'PlayTennis', features)

print("Decision Tree using ID3 Algorithm:")
print(decision_tree)


# In[ ]:





# In[ ]:




