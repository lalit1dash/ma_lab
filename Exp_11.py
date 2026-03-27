#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Implement SVM for a dataset and compare the accuracy by applying the following kernel functions (i)Linear (ii)Polynomial (iii)RBF 


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[4]:


df=pd.read_csv("employee_turnover.csv")
df.head() 


# In[5]:


x=df.drop("Employee_Turnover",axis=1)
y=df["Employee_Turnover"]


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[7]:


kernels=['linear','poly','rbf']
results={ } 


# In[8]:


for kernel in kernels:
    clf=SVC(kernel=kernel,C=1.0,gamma='scale') 
    clf.fit(x_train,y_train) 
    y_pred=clf.predict(x_test) 
    results[kernel]=accuracy_score(y_test,y_pred)


# In[9]:


for kernel,accuracy in results.items():
    print(f"Kernel:{kernel}, Accuracy:{accuracy:.4f}")


# In[10]:


import matplotlib.pyplot as plt  
plt.plot(list(results.keys()), list(results.values()),marker='o') 
plt.xlabel("Kernels")
plt.ylabel("Accuracy")
plt.title("Kernels VS accuracy")
plt.show()


# In[11]:


plt.figure(figsize=(8, 6))
plt.bar(list(results.keys()), list(results.values()), color=['orange', 'indigo', 'green'])
plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.title('SVM Kernel Comparison by Accuracy on Iris Dataset')
plt.ylim(0.0, 1.5) # Accuracy is between 0 and 1
plt.show() 


# In[ ]:





# In[ ]:




