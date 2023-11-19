#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv('./diabetes.csv')
data.head()


# In[4]:


#Check for null or missing values
data.isnull().sum()


# In[5]:


#Replace zero values with mean values
for column in data.columns[1:-3]:
    data[column].replace(0, np.NaN, inplace = True)
    data[column].fillna(round(data[column].mean(skipna=True)), inplace = True)
data.head(10)


# In[6]:


X = data.iloc[:, :8] #Features
Y = data.iloc[:, 8:] #Predictor


# In[7]:


#Perform Spliting
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[8]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_fit = knn.fit(X_train, Y_train.values.ravel())
knn_pred = knn_fit.predict(X_test)


# In[9]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
print("Confusion Matrix")
print(confusion_matrix(Y_test, knn_pred))
print("Accuracy Score:", accuracy_score(Y_test, knn_pred)*100)
print("Reacal Score:", recall_score(Y_test, knn_pred)*100)
print("F1 Score:", f1_score(Y_test, knn_pred)*100)
print("Precision Score:",precision_score(Y_test, knn_pred)*100)

