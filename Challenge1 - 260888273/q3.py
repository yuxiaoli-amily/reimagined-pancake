#!/usr/bin/env python
# coding: utf-8

# In[10]:


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

# load the data
dataset = load_breast_cancer() 


# In[11]:


# get the index of features that we want 
index1 = np.where(dataset['feature_names'] == 'mean concavity')
index2 = np.where(dataset['feature_names'] == 'worst area')
index1 = index1[0][0]
index2 = index2[0][0]

# store the data of features independently
mean_concavity = dataset.data[:, index1]
worst_area = dataset.data[:, index2]

# divide into two groups based on their labels
mean_concavity_zero = []
mean_concavity_one = []
worst_area_zero = []
worst_area_one = []

for i in range(len(dataset.target)):
    if dataset.target[i] == 0:
        mean_concavity_zero.append(mean_concavity[i])
        worst_area_zero.append(worst_area[i])
    else:
        mean_concavity_one.append(mean_concavity[i])
        worst_area_one.append(worst_area[i])
        

# plotting 
l1 = plt.scatter(mean_concavity_zero, worst_area_zero, color='r')
l2 = plt.scatter(mean_concavity_one, worst_area_one, color='g')
plt.legend(handles=[l1,l2],labels=['malignant','benign'],loc='best')
plt.xlabel('Mean Concavity')
plt.ylabel('Worst Area')
plt.show()


# In[ ]:




