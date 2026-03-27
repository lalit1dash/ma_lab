#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Consider a dataset,use random forest to predict the output class.vary the number of trees as follows and compare the results (i)20(ii)50(iii)100(iv)200(v)500


# In[35]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 


# In[36]:


df=pd.read_csv("employee_turnover.csv")
df.head() 


# In[37]:


x=df.drop("Employee_Turnover",axis=1)
y=df["Employee_Turnover"]


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[39]:


n_trees=[20,50,100,200,500]
accuracy_list=[] 
for i in n_trees:
     rf_classifier = RandomForestClassifier(n_estimators=i, random_state=42)    
     rf_classifier.fit(x_train, y_train)  
     y_pred = rf_classifier.predict(x_test)
     accuracy_list.append(accuracy_score(y_test,y_pred)) 


# In[40]:


print(accuracy_list)  


# In[41]:


import matplotlib.pyplot as plt
plt.plot(n_trees,accuracy_list,marker='o')      
plt.xlabel("Value of n_trees")    
plt.ylabel("Accuracy")
plt.title("accuracy VS n_trees")
plt.show() 


# In[ ]:




