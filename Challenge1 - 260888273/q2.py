#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[7]:


## Method One
# store features seperately
features = []
n = 0
while n < len(dataset.data[1]):
    feature = []
    for i in range(len(dataset.data)):
        feature.append(dataset.data[i][n])
    n = n+1
    features.append(feature)

# calculate the correlation in pairs
rela = []
for i in range(len(features)):
    n = i+1
    while n < len(features):
        relation = np.corrcoef(features[i],features[n])
        rela.append(relation[0][1])
        n = n+1

# plotting        
plt.hist(rela,bins = 50)
plt.xlabel('Pearson Correlation')
plt.ylabel('Counts')
plt.title('Pearson Correlation Between Pair of Features')
plt.show


# In[8]:


# Method Two

# get the correlation matrix directly and keep the lower-diagonal only
matrix = np.corrcoef(dataset.data, rowvar = False)
lower_diagonal = np.tril(matrix, k = 1)
coef = lower_diagonal.flatten()

# update the coef to be non-zeros
coef = coef[coef != 0]

# plotting
plt.hist(coef, bins = 50)
plt.xlabel('Pearson Correlation')
plt.ylabel('Counts')
plt.title('Pearson Correlation Between Pair of Features')
plt.show


# In[ ]:




