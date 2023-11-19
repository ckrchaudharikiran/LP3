#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv(r"C:\Users\kchau\Desktop\lp3\First-main\uber.csv")
df.info()


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.isnull()


# In[7]:


df.drop(columns=["Unnamed: 0", "key"], inplace=True)
df.head()


# In[8]:


df['dropoff_latitude'].fillna(value=df['dropoff_latitude'].mean(),inplace = True)
df['dropoff_longitude'].fillna(value=df['dropoff_longitude'].median(),inplace = True)


# In[9]:


df.dtypes


# In[10]:


df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
df.dtypes


# In[11]:


df = df.assign(hour = df.pickup_datetime.dt.hour,
               day = df.pickup_datetime.dt.day,
               month = df.pickup_datetime.dt.month,
               year = df.pickup_datetime.dt.year,
               dayofweek = df.pickup_datetime.dt.dayofweek)


# In[12]:


df


# In[13]:


df = df.drop(["pickup_datetime"], axis =1)
df


# In[14]:


from math import *
    
def distance_formula(longitude1, latitude1, longitude2, latitude2):
    travel_dist = []
    
    for pos in range (len(longitude1)):
        lon1, lan1, lon2, lan2 = map(radians, [longitude1[pos], latitude1[pos], longitude2[pos], latitude2[pos]])
        dist_lon = lon2 - lon1
        dist_lan = lan2 - lan1
        
        a = sin(dist_lan/2)**2 + cos(lan1) * cos(lan2) * sin(dist_lon/2)**2
        
        #radius of earth = 6371
        c = 2 * asin(sqrt(a)) * 6371 
        travel_dist.append(c)
            
    return  travel_dist


# In[16]:


df['dist_travel_km'] = distance_formula(df.pickup_longitude.to_numpy(), df.pickup_latitude.to_numpy(), df.dropoff_longitude.to_numpy(), df.dropoff_latitude.to_numpy())


# In[17]:


df.plot(kind = "box",subplots = True,layout = (6,2),figsize=(15,20)) #Boxplot to check the outliers
plt.show()


# In[18]:


#Using the InterQuartile Range to fill the values
def remove_outlier(df1 , col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1-1.5*IQR
    upper_whisker = Q3+1.5*IQR
    df[col] = np.clip(df1[col] , lower_whisker , upper_whisker)
    return df1

def treat_outliers_all(df1 , col_list):
    for c in col_list:
        df1 = remove_outlier(df , c)
    return df1


# In[19]:


df = treat_outliers_all(df , df.iloc[: , 0::])


# In[20]:


df.plot(kind = "box",subplots = True,layout = (7,2),figsize=(15,20)) 
plt.show()


# In[21]:


corr = df.corr() 
corr


# In[22]:


fig,axis = plt.subplots(figsize = (10,6))
sns.heatmap(df.corr(),annot = True) #Correlation Heatmap (Light values means highly correlated)


# In[23]:


df_x = df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','month','year','dayofweek','dist_travel_km']]
df_y = df['fare_amount']


# In[24]:


# Dividing the dataset into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=1)


# In[25]:


df


# In[26]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train, y_train)


# In[27]:


y_pred_lin = reg.predict(x_test)
print(y_pred_lin)


# In[28]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)
rf.fit(x_train,y_train)


# In[29]:


y_pred_rf = rf.predict(x_test)
print(y_pred_rf)


# In[30]:


cols = ['Model', 'RMSE', 'R-Squared']

result_tabulation = pd.DataFrame(columns = cols)


# In[31]:


from sklearn import metrics 
from sklearn.metrics import r2_score 

reg_RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lin))
reg_squared = r2_score(y_test, y_pred_lin)

full_metrics = pd.Series({'Model': "Linear Regression", 'RMSE' : reg_RMSE, 'R-Squared' : reg_squared})

result_tabulation = result_tabulation.append(full_metrics, ignore_index = True)

result_tabulation


# In[33]:


rf_RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))
rf_squared = r2_score(y_test, y_pred_rf)


full_metrics = pd.Series({'Model': "Random Forest ", 'RMSE':rf_RMSE, 'R-Squared': rf_squared})
result_tabulation = result_tabulation.append(full_metrics, ignore_index = True)

result_tabulation


# In[ ]:




