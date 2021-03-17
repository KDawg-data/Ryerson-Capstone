#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


## Load Data into dataframe


# In[3]:


df = pd.read_csv('Capstone.csv')


# In[4]:


df


# # Group By Age

# In[5]:


#Create new column for age groups, and populate based on following logic:
    #40 - 64 = Adult
    #65 - 79 = Senior
    #80+ = Senior+
bins= [40, 65, 80, 120]
labels = ['Adult', 'Senior', 'Senior+']
df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

df = df[['age', 'AgeGroup',
 'anaemia',
 'creatinine_phosphokinase',
 'diabetes',
 'ejection_fraction',
 'high_blood_pressure',
 'platelets',
 'serum_creatinine',
 'serum_sodium',
 'sex',
 'smoking',
 'time',
 'DEATH_EVENT'
 ]]

df


# # Distribution of Variables

# In[ ]:


'''Create box plots for each numeric variable '''
col_names = list(df.columns.values) # make list with all variables
col_names_numeric = col_names.copy() # make copy of original column list

#make list of non-numeric variables
non_numeric_variables = ['AgeGroup', 'anaemia','diabetes', 'high_blood_pressure', 'sex', 'smoking']

#remove all non-numeric variables from the original list
col_names_numeric = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']

#print box plot for each numeric variables
for i in col_names_numeric:
    df.boxplot(column=i)
    plt.show()


# In[ ]:


#Create boxplots grouped by Age Group
for i in col_names_numeric:
    df.boxplot(column=i, by='AgeGroup', figsize = (10,10))
    plt.show()


# In[ ]:


#Create histogram for all variables (numeric and non-numeric)

df.hist(figsize=(10,10))


# In[ ]:


#Comment on Distribution of Each Numeric Variable
'''
Normal Distribution/ Slight Skew:

    Age: all patients in this dataset are above 40, with the majority around 50 - 70. Slightly right skewed
    Ejection_Fraction(EF): slightly right skew
    Serum_sodium: slight left skew

High number of outliers:

    Creatine_Phosphate (CP): the boxplot indicates this varaible has many outliers from 1500 - 8000. Right skewed
    Serum_Creatine (SC): this variable has many outliers beyond the expected range of 2. Right skewed
    Platelets: this varible also shows a high number of outliers, outside the expected range of 150000 - 400000. Right skew
    
'''


# # Summary Statistics

# In[ ]:


#Get summary statistics about dataset using describe function

df.describe()


# In[ ]:


#Generate correlation matrix from dataset to identify highly correlated variables to death
corrMatrix = df.corr('spearman')

corrMatrix.sort_values(by='DEATH_EVENT', ascending = False)


# In[ ]:


'''
Weakly correlated with death: Diabetes, sex, smoking, platelets

Strongly correlated with death: serum sodium, ejection fraction, serum creatine, age'''


# # Modeling

# In[17]:


#Create Test and Training Sets

# Choose target and features
y = df.DEATH_EVENT
df_features = ['age','anaemia', 'creatinine_phosphokinase', 'diabetes',
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


#Create Classifer Decision Tree Model - all attributes, test train split


# In[18]:


#Modeling
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

 
#Making the model
class_model = RandomForestClassifier(random_state=1)
 
#Fitting the model
class_model.fit(train_X, train_y)
 
#testing the model
df_preds = class_model.predict(val_X)
score = accuracy_score(val_y, df_preds)
print(score)


# In[16]:


k = 10
kf = KFold(n_splits=k, random_state=None)
class_model = RandomForestClassifier(random_state=1)
 
acc_score = []
 
for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    class_model.fit(X_train,y_train)
    pred_values = class_model.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)
     
avg_acc_score = sum(acc_score)/k
print(avg_acc_score)


# In[ ]:


'''
Summary of Model Performance

    Attributes = all
    Model = Random Forest Classifer

    The k fold model showed 83.3% accuracy
    Train test split model showed 85.3% accuracy
'''


# In[ ]:


#Create Classifer Logistic Regression Model - all attributes


# In[19]:


from sklearn.linear_model import LogisticRegression


#Making the model
logisticRegr = LogisticRegression(random_state=1)
 
#Fitting the model
logisticRegr.fit(train_X, train_y)
 
#testing the model
df_log_preds = logisticRegr.predict(val_X)
score = accuracy_score(val_y, df_log_preds)
print(score)


# In[ ]:


#Create Classifer Logistic Regression Model - all attributes using KFold


# In[13]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

k = 10
kf = KFold(n_splits=k, random_state=None)
model = LogisticRegression(solver= 'liblinear')
 
acc_score = []
 
for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)
     
avg_acc_score = sum(acc_score)/k
print(avg_acc_score)


# In[ ]:


'''
Summary of Model Performance
    Attributes = all
    Model = Logistic Regression

    The k-fold model showed 79.3% accuracy
    Train-test model showed 74.6% accuracy
'''

