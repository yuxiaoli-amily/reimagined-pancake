#!/usr/bin/env python
# coding: utf-8

# In[1]:


# installing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# load the data
dataset = load_breast_cancer() 


# In[2]:


# create the pipeline model with cross-validator
kf = KFold(n_splits = 5, shuffle = False)
model = make_pipeline(StandardScaler(), LogisticRegression())

X = dataset.data
y = dataset.target
acc_score = []

# modelling on each training split
for train_index , test_index in kf.split(X):
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)   
    acc_score.append(acc)

# plotting    
number_of_split = ['1', '2', '3', '4', '5']
plt.scatter(number_of_split, acc_score)
plt.xlabel('Split Time')
plt.ylabel('Acurracy')
plt.title('Scatterplot of the Test Accuracies for each Split')
plt.show


# In[ ]:




