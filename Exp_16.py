#!/usr/bin/env python
# coding: utf-8

# In[1]:


# implement Naive Bayes Classification in python 


# In[5]:


import math
import numpy as np

class NaiveBayesFromScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.stats = {}

        for c in self.classes:
            X_c = X[y == c]
            # Convert class to string to avoid UFuncTypeError when creating dictionary keys
            class_key = str(c)

            # Store Mean and Std Dev for every feature
            self.stats[class_key] = [(np.mean(feature), np.std(feature)) for feature in X_c.T]

            # Store the Prior Probability P(Class)
            self.stats[class_key + "_prior"] = len(X_c) / len(X)

    def calculate_gaussian_probability(self, x, mean, std):
        # Adding a tiny epsilon to std to prevent DivisionByZero if std is 0
        epsilon = 1e-6
        std = std + epsilon
        exponent = math.exp(-((x - mean) ** 2 / (2 * (std ** 2))))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        posteriors = []

        for c in self.classes:
            class_key = str(c)
            prior = self.stats[class_key + "_prior"]
            likelihood = 1

            for i, (mean, std) in enumerate(self.stats[class_key]):
                likelihood *= self.calculate_gaussian_probability(x[i], mean, std)

            posteriors.append((prior * likelihood, c))

        # Return the class (c) associated with the highest probability
        return max(posteriors, key=lambda item: item[0])[1]

# --- Test Data ---
X = np.array([[150, 7], [170, 8], [140, 6], [130, 5], [200, 9], [210, 10]])
y = np.array([0, 0, 0, 1, 1, 1])

model = NaiveBayesFromScratch()
model.fit(X, y)

# Predict for a new fruit
new_fruit = np.array([[160, 7.5]])
prediction = model.predict(new_fruit)

print(f"Prediction for {new_fruit[0]}: {'Apple' if prediction[0] == 0 else 'Orange'}")

