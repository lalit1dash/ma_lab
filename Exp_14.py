#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Develope logistic Regression Model for a given Dataset 


# In[11]:


import pandas as pd 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,classification_report


# In[4]:


df=pd.read_csv("employee_turnover.csv")
df


# In[5]:


x=df.drop("Employee_Turnover",axis=1)
y=df["Employee_Turnover"]


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[15]:


model=LogisticRegression(max_iter=200) 
model.fit(x_train,y_train)


# In[16]:


y_pred=model.predict(x_test)


# In[17]:


print("Accuracy_ Score:",accuracy_score(y_test,y_pred)) 
print("Classification_report:",classification_report(y_test,y_pred))


# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("employee_turnover.csv")

# Features and target
X = data.drop("Employee_Turnover", axis=1)
y = data["Employee_Turnover"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot predicted vs actual
plt.scatter(range(len(y_test)), y_test, label="Actual")
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker='x')

plt.xlabel("Test Samples")
plt.ylabel("Employee Turnover (0 or 1)")
plt.title("Actual vs Predicted Turnover")
plt.legend()
plt.show()


# In[ ]:




