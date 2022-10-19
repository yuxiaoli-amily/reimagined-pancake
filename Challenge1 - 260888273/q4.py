#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# create a list to store features
features = []
n = 0
while n < len(dataset.data[1]):
    feature = []
    for i in range(len(dataset.data)):
        feature.append(dataset.data[i][n])
    n = n+1
    features.append(feature)  

# calculate the mean/median/variance for each feature    
mean_list = []
median_list = []
variance_list = []

for i in range(len(features)):
    mean_list.append(np.mean(features[i]))
    median_list.append(np.median(features[i]))
    variance_list.append(np.std(features[i]))

# find the maximum 
maxMean = np.amax(mean_list)
maxMedian = np.amax(median_list)
maxVariance = np.amax(variance_list)

index1 = np.where(mean_list == maxMean)[0][0]
index2 = np.where(median_list == maxMedian)[0][0]
index3 = np.where(variance_list == maxVariance)[0][0]


highest_mean_feature = features[index1]
highest_median_feature = features[index2]
highest_variance_feature = features[index3]


# combine these different collections into a list
data_to_plot = [features[index1], features[index2], features[index3]]

# plotting
plt.violinplot(data_to_plot, showextrema=False)
plt.xlabel('Value of Features')
plt.ylabel('Features')
plt.title('Violin Plot of Features with Higest Mean, Median and Variance')
plt.show()


# In[ ]:




