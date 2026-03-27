#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Build Knn classification Model for the given dataset vary the number of k Values as follows and compare the results (i)1 (ii)3 (iii)5 (iv)7 (v)11 


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score,classification_report 


# In[4]:


df=pd.read_csv("employee_turnover.csv")
df.head() 


# In[6]:


x=df.drop("Employee_Turnover",axis=1)
y=df["Employee_Turnover"]


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)   


# In[8]:


model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train) 


# In[9]:


y_pred=model.predict(x_test)


# In[15]:


print("Accuracy: ",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred)) 


# In[11]:


n=[1,3,5,7,11]
accuracy_list=[] 
for k in n:
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train) 
    y_pred=model.predict(x_test)
    accuracy_list.append(accuracy_score(y_test,y_pred))


# In[12]:


plt.plot(n,accuracy_list,marker='o')
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.title("KNN accuracy VS K")
plt.show() 


# In[13]:


print(accuracy_list)


# In[ ]:




