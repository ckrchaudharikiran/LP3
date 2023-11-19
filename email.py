#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import metrics


# In[3]:


df = pd.read_csv(r"C:\Users\kchau\Downloads\emails.csv\emails.csv")
df


# In[5]:


df.shape


# In[6]:


df.isnull().any()


# In[7]:


df.drop(columns='Email No.', inplace=True)
df


# In[8]:


df.columns


# In[10]:


df.Prediction.unique()


# In[11]:


df['Prediction'] = df['Prediction'].replace({0:'Not spam', 1:'Spam'})


# In[12]:


df


# In[13]:


X = df.drop(columns='Prediction',axis = 1)
Y = df['Prediction']


# In[16]:


X.columns


# In[17]:


Y.head()


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# In[19]:


KN = KNeighborsClassifier
knn = KN(n_neighbors=7)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)


# In[20]:


print("Prediction: \n")
print(y_pred)


# In[21]:


# Accuracy

M = metrics.accuracy_score(y_test,y_pred)
print("KNN accuracy: ", M*100)


# In[22]:


C = metrics.confusion_matrix(y_test,y_pred)
print("Confusion matrix: ", C)


# In[23]:


model = SVC(C = 1)   # cost C = 1

model.fit(x_train, y_train)

y_pred = model.predict(x_test)      # predict


# In[24]:


kc = metrics.confusion_matrix(y_test, y_pred)
print("SVM Confution Matrix: \n", kc)
svma = metrics.accuracy_score(y_test,y_pred)
print("SVM accuracy: ", svma*100)


# In[ ]:




