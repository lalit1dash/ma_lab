#!/usr/bin/env python
# coding: utf-8

# In[27]:


# wap to demonstrate the working of the decision tree based ID3 algorithm. use an approptiate dataset for building the decesion tree and apply this knowlwdge to classify a new sample


# In[28]:


import seaborn as sns
import pandas as pd
import math

# Load Titanic dataset
df = sns.load_dataset('titanic')

print(df.head())


# In[29]:


df = df[['survived', 'pclass', 'sex', 'age', 'embarked']]

df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

df['age'] = pd.cut(df['age'],
                   bins=[0,12,19,40,60,100],
                   labels=['Child','Teen','Adult','Middle','Senior'])

df['survived'] = df['survived'].map({0:'No',1:'Yes'})


# In[30]:


import math

def entropy(col):
    probs = col.value_counts(normalize=True)
    return -sum([p * math.log2(p) for p in probs])

def info_gain(data, feature, target="survived"):
    total_entropy = entropy(data[target])

    weighted_entropy = 0
    for v in data[feature].unique():
        subset = data[data[feature] == v]
        weighted_entropy += (len(subset)/len(data)) * entropy(subset[target])

    return total_entropy - weighted_entropy

def id3(data, target="survived", features=None):
    if features is None:
        features = data.columns.drop(target)

    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    if len(features) == 0:
        return data[target].mode()[0]

    gains = [info_gain(data, f, target) for f in features]
    best_feature = features[gains.index(max(gains))]

    tree = {best_feature: {}}

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = id3(subset, target, features.drop(best_feature))
        tree[best_feature][value] = subtree

    return tree

tree = id3(df)
print(tree)


# In[31]:


new_sample = {
    'pclass': 3,
    'sex': 'male',
    'age': 'Adult',
    'embarked': 'S'
}


# In[32]:


def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    root = list(tree.keys())[0]
    value = sample[root]

    return predict(tree[root][value], sample)

print(predict(tree, new_sample))


# In[ ]:




