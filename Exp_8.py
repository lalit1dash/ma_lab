#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score,classification_report 


# In[6]:


iris=load_iris()

x=iris.data
y=iris.target


# In[15]:


y


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)   


# In[10]:


model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train) 


# In[16]:


y_pred=model.predict(x_test)


# In[20]:


print("Accuracy: ",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[21]:


model=KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train) 


# In[22]:


y_pred=model.predict(x_test)


# In[35]:


print("Accuracy: ",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred,target_names=iris.target_names)) 


# In[45]:


accuracy_list=[] 
for k in range(1,21):
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train) 
    y_pred=model.predict(x_test)
    accuracy_list.append(accuracy_score(y_test,y_pred))


# In[46]:


plt.plot(range(1,21),accuracy_list,marker='o')
plt.xlabel("Value of K")
plt.ylabel("Accuracy")     
plt.title("KNN accuracy VS K")
plt.show() 


# In[44]:


print(accuracy_list)

