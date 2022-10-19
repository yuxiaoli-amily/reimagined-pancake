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


# split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, shuffle = False)

# modelling 
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(x_train, y_train)

# the out-of-sample classification accuracy
score = model.score(x_test, y_test)
print(score)

# get the number of iterations needed for convergence
print(model.named_steps['logisticregression'].n_iter_)


# In[ ]:




