#!/usr/bin/env python
# coding: utf-8

# In[157]:


# Develope a program to demonastrate  the working of the decision tree for building use breast canser dataset the decision tree and apply this knowledge to classify a new sample. 


# In[158]:


# 🔹 Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 🔹 Step 2: Load dataset
df = pd.read_csv("Breast_Cancer.csv")

# 🔹 Step 3: Clean column names
df.columns = df.columns.str.strip()

# 🔹 Step 4: Clean string values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()

# 🔹 Step 5: Encode categorical data
encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# 🔹 Step 6: Split features & target
X = df.drop("Status", axis=1)
y = df["Status"]

# 🔹 Step 7: Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Step 8: Train Decision Tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
model.fit(x_train, y_train)

# 🔹 Step 9: Evaluate model
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔹 Step 10: Predict a new sample
new_sample = x_test.iloc[0:1]
prediction = model.predict(new_sample)

# 🔹 Step 11: Decode prediction (important)
decoded = encoders["Status"].inverse_transform(prediction)

print("\nEncoded Prediction:", prediction)
print("Actual Meaning:", decoded)


# In[160]:

import matplotlib.pyplot as plt
from sklearn import tree
plt.figure(figsize=(15,8))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=encoders["Status"].classes_,
    filled=True,
    max_depth=4
)
plt.title("Decision Tree Visualization")
plt.show()


# In[ ]:




