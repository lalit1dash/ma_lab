#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


#Creation an loading different types of datasets in python using the required libraries.
#i creation using pandas 
import pandas as pd

data = {
    "Name": ["Biswa", "Lalit", "Ashu"],   
    "Age": [25, 30, 35],
    "City": ["New York", "London", "Paris"]
}

df = pd.DataFrame(data)
print(df)

data = [
    ["Biswa", 25, "New York"],
    ["Lalit", 30, "London"],
    ["Ashu", 35, "Paris"]
]

df = pd.DataFrame(data, columns=["Name", "Age", "City"])
print(df)

df.to_csv("sample_data.csv", index=False) 


# In[4]:


#ii loading csv dataset files using pandas 
df=pd.read_csv("sample_data.csv")   
print(df)


# In[8]:


#iii loading dataset using sklearn  
from sklearn.datasets import load_digits, load_wine, load_breast_cancer

digits = load_digits()
wine = load_wine()
cancer = load_breast_cancer() 

print(digits)
print(wine) 
print(cancer)


# In[10]:


#Q2 write a program to compute mean,median,mode,varience,standard deviation using datasets.
import pandas as pd

data = {
    "Marks": [65, 70, 75, 80, 85, 90, 75, 80]
}

df = pd.DataFrame(data)

values = df["Marks"]

mean = values.mean()
median = values.median()
mode = values.mode()[0]        
variance = values.var()
std_deviation = values.std()

print("Dataset:")
print(values)

print("\nStatistical Measures:")
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Variance:", variance)
print("Standard Deviation:", std_deviation)




# In[11]:


#Q3 Demonstrate various data pre-processing technique for a given dataset.Write a python program to compute 
#i reshaping the data
#ii filtering the data 
#iii merging the data 
#iv Handling the missing values in dataset 
#v feature normalization : min-max normalization 

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- SETTING UP SAMPLE DATA ---
data1 = {
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Score': [85, np.nan, 70, 95, 80], # Contains a missing value
    'Age': [25, 30, 35, 40, 45]
}
data2 = {
    'ID': [1, 2, 3, 4, 5],
    'City': ['NY', 'LA', 'CHI', 'HOU', 'PHX']
}

df = pd.DataFrame(data1)
df_extra = pd.DataFrame(data2)

# i. RESHAPING THE DATA (Melting)
# Converting wide data to long format
df_reshaped = pd.melt(df, id_vars=['ID', 'Name'], value_vars=['Score', 'Age'], 
                      var_name='Metric', value_name='Value')

# ii. FILTERING THE DATA
# Extracting rows where Age is greater than 30
df_filtered = df[df['Age'] > 30]

# iii. MERGING THE DATA
# Combining two dataframes based on a common key ('ID')
df_merged = pd.merge(df, df_extra, on='ID')

# iv. HANDLING MISSING VALUES
# Filling the missing Score with the mean of the column
df['Score'] = df['Score'].fillna(df['Score'].mean())

# v. FEATURE NORMALIZATION (Min-Max Scaling)
# Rescaling Age to a range between 0 and 1
scaler = MinMaxScaler()
df['Age_Normalized'] = scaler.fit_transform(df[['Age']])

# --- OUTPUTS ---
print("--- Merged & Cleaned Data ---")
print(df_merged)
print("\n--- Handled Missing Values & Normalized Age ---")
print(df)
print("\n--- Reshaped (Melted) Data ---")
print(df_reshaped.head(4))

