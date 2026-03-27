#!/usr/bin/env python
# coding: utf-8



import pandas as pd 




#Q1 create a series using pandas and display.
s=pd.Series([10,20,30,40,50])
s


# In[4]:


#Q2 Access the index and values of our series.
s=pd.Series([11,22,33,44,55]) 
print("index:",s.index)
print("values:",s.values)


# In[5]:


#Q3 compare an array using numpy with series using pandas. 
import numpy as np
import pandas as pd 

arr=np.array([1,2,3,4,5])
print(arr[2]) 
series=pd.Series([1,2,3,4,5],index=['A','B','C','D','E']) 
print(series['C'])


# In[6]:


#Q4 Define series objects with indivisual indices  
data=['Apple','Banana','Guava','Orange','mango']   
ind=['A','B','C','D','E'] 
s=pd.Series(data,index=ind)
s


# In[7]:


#Q5 Access single values of a series.
s=pd.Series([11,22,33,44,55])
s[2]


# In[9]:


#Q6 Load datasets in a dataframe variable using pandas.
df=pd.read_csv("data.csv")
df


# In[30]:


#Q7 Usage of different methods in matplotlib. 
import matplotlib.pyplot as plt 

x=[1,2,3,4,5]
y=[20,24,36,19,25] 
z=[1,2,2,3,3,3,4,4,5,5,5,5]

#plt.plot(x,y)
#plt.scatter(x,y)
#plt.bar(x,y) 
plt.hist(z)
#plt.pie(y,autopct='%1.1f%%')   
plt.show() 


# In[ ]:




