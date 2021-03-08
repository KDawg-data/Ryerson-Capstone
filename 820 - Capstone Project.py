#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


## Load Data into dataframe


# In[4]:


df = pd.read_csv('Capstone.csv')


# In[5]:


df


# # Summary Statistics

# In[6]:


#Get summary statistics about dataset using describe function

df.describe()


# In[7]:


#Generate correlation matrix from dataset to identify highly correlated variables to death

corrMatrix = df.corr()
corrMatrix


# In[8]:


corrMatrix.sort_values(by='DEATH_EVENT', ascending = False)


# In[9]:


import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(corrMatrix, annot=True)
plt.show()


# In[10]:


#Weakly correlated with death: Diabetes, sex, smoking, platelets
#Strongly correlated with death: serum sodium, ejection fraction, serum creatine, age


# In[16]:


#Create Test and Training Sets

# Choose target and features
y = df.DEATH_EVENT
df_features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
X = df[df_features]

from sklearn.model_selection import train_test_split
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


# In[ ]:


#Create Decision Tree Model - all attributes


# In[17]:


#Modeling
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
 
#Making the model
forest_model = RandomForestRegressor(random_state=1)
 
#Fitting the model
forest_model.fit(train_X, train_y)
 
#testing the model
df_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, df_preds))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




