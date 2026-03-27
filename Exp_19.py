#!/usr/bin/env python
# coding: utf-8

# In[1]:


# wap to implement k_nearest neighbour algorithm to classify the iris dataset. print both correct and wrong predictionjava/python ML.


# In[3]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[4]:


# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names


# In[5]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# In[8]:


# Create KNN model
k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train,y_train)


# In[9]:


# Predict
y_pred = model.predict(X_test)


# In[10]:


# ---------------- OUTPUT ---------------- #

correct = 0
wrong = 0

print("\n--- Predictions ---\n")

for i in range(len(y_test)):
    actual = target_names[y_test[i]]
    predicted = target_names[y_pred[i]]

    if y_test[i] == y_pred[i]:
        print(f"Sample {i+1}: Correct → Actual: {actual}, Predicted: {predicted}")
        correct += 1
    else:
        print(f"Sample {i+1}: Wrong → Actual: {actual}, Predicted: {predicted}")
        wrong += 1


# In[12]:


# Accuracy
accuracy = correct / len(y_test)

print("\nCorrect:", correct)
print("Wrong:", wrong)
print("Accuracy:", accuracy)


# In[ ]:




