#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Develope Decesion tree classification model for a given dataset and use it to classify a new sample 


# In[3]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


# In[4]:


titanic=sns.load_dataset("titanic")


# In[5]:


titanic.head()


# In[6]:


titanic.isnull().sum()


# In[7]:


features=["pclass","sex","fare","embarked","age"]  
target=["survived"]


# In[8]:


from sklearn.impute import SimpleImputer 
imp_median=SimpleImputer(strategy="median")
titanic[["age"]]=imp_median.fit_transform(titanic[["age"]]) 

imp_freq=SimpleImputer(strategy="most_frequent")
titanic[["embarked"]]=imp_freq.fit_transform(titanic[["embarked"]]) 


# In[9]:


from sklearn.preprocessing import LabelEncoder 

le=LabelEncoder() 
titanic["sex"]=le.fit_transform(titanic["sex"]) 
titanic["embarked"]=le.fit_transform(titanic["embarked"]) 


# In[10]:


x=titanic[features]
y=titanic[target] 


# In[11]:


x


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[13]:


from sklearn.tree import DecisionTreeClassifier 
model=DecisionTreeClassifier() 
model.fit(x_train,y_train)


# In[14]:


y_pred=model.predict(x_test)


# In[15]:


from sklearn.metrics import accuracy_score 
print("Accuracy:",accuracy_score(y_test,y_pred)) 


# In[17]:


from sklearn.tree import plot_tree

plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=x.columns,
    class_names=["Died", "Survived"],
    filled=True,
    max_depth=3
)

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




