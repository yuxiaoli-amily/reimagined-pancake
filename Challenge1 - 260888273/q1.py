#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


# transform the data
trans_data = PCA(n_components=2).fit_transform(dataset['data'])

# plotting
plt.scatter(trans_data[:,0], trans_data[:,1])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Two Components PCA')
plt.show


# In[ ]:




