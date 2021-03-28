#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np

#Plotting
import matplotlib.pyplot as plt
import seaborn as sn

#Modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import statsmodels.api as sm

#SK Metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[123]:


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


# In[6]:


#Converting Age Group from Cateogrical to Numerical Using Label Encoder


# In[7]:


# creating instance of labelencoder
labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column
df['AgeGroup_Cat'] = labelencoder.fit_transform(df['AgeGroup'])

df = df[['age', 'AgeGroup', 'AgeGroup_Cat',
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

# In[8]:


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


# In[9]:


#Create boxplots grouped by Age Group
for i in col_names_numeric:
    df.boxplot(column=i, by='AgeGroup', figsize = (10,10))
    plt.show()


# In[10]:


#Create histogram for all variables (numeric and non-numeric)

df.hist(figsize=(10,10))


# In[11]:


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

# In[12]:


#Get summary statistics about dataset using describe function

df[['age',
 'anaemia',
 'creatinine_phosphokinase',
 'ejection_fraction',
 'high_blood_pressure',
 'platelets',
 'serum_creatinine',
 'serum_sodium',
 'time']].describe()


# In[13]:


#Generate correlation matrix from dataset to identify highly correlated variables to death
corrMatrix = df.corr('spearman')

corrMatrix.sort_values(by='DEATH_EVENT', ascending = False)


# In[14]:


'''
Weakly correlated with death: Diabetes, sex, smoking, platelets

Strongly correlated with death: serum sodium, ejection fraction, serum creatine, age'''


# # Target and Features

# In[15]:


# Choose target and features
y = df.DEATH_EVENT
df_features = ['age','AgeGroup_Cat','anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
X = df[df_features]


# In[16]:


#Removed Diabetes, Smoking, Sex, and Platelets

df_features_lim = ['AgeGroup_Cat','anaemia',
       'ejection_fraction', 'high_blood_pressure',
       'serum_creatinine', 'serum_sodium', 'time']
X_lim = df[df_features_lim]


# # Create Classifer RFC Model - all attributes

# In[173]:


#Making the model
class_model = RandomForestClassifier(random_state=None)
 
#Fitting the model
class_model.fit(X, y)


#Cross Validation - 10
k = 10
kf = KFold(n_splits=k, random_state=None)

result = cross_val_score(class_model , X, y, cv = kf)
 
print("Avg accuracy: {}".format(result.mean()))


# # Confusion Matrix - DT

# In[189]:


#Confusion Matrix

#Create Test-Train Split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

#Create Model
class_model_cm = RandomForestClassifier(random_state=None)
class_model_cm.fit(train_X, train_y)

#Predict
df_preds = class_model_cm.predict(val_X)


confusion_matrix = pd.crosstab(val_y, df_preds, rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()

print(classification_report(val_y, df_preds))


# In[ ]:


'''
Summary of Model Performance

    Attributes = all
    Model = Random Forest Classifer

    The k fold model showed 83.6% accuracy
'''


# # Create Classifer RFC Model - Limited Attributes

# In[184]:


#Making the model
class_model_lim = RandomForestClassifier(random_state=None)

#Fitting the model

class_model_lim.fit(X_lim, y)


#Cross Validation - 10
k = 10
kf = KFold(n_splits=k, random_state=None)

result = cross_val_score(class_model_lim, X_lim, y, cv = kf)

print("Avg accuracy: {}".format(result.mean()))


# In[185]:


#Confusion Matrix

#Create Test-Train Split
train_X, val_X, train_y, val_y = train_test_split(X_lim, y,random_state = None)

#Create Model
class_model_lim_cm = RandomForestClassifier(random_state=None)

class_model_lim_cm.fit(train_X, train_y)

#Predict
df_preds = class_model_lim_cm.predict(val_X)


confusion_matrix = pd.crosstab(val_y, df_preds, rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()


# # Create Classifer Logistic Regression Model - all attributes

# In[177]:


#Making the model
log_model = LogisticRegression(solver= 'liblinear')
 
#Fitting the model
log_model.fit(X, y)

#Cross Validation - 10
k = 10
kf = KFold(n_splits=k, random_state=None)
result = cross_val_score(log_model , X, y, cv = kf)
 
print("Avg accuracy: {}".format(result.mean()))


# # Confusion Matrix - LR

# In[191]:


#Confusion Matrix

#Create Test-Train Split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state =None)

#Create Model
log_model_cm = LogisticRegression(solver= 'liblinear')

log_model_cm.fit(train_X, train_y)

#Predict
df_preds = log_model_cm.predict(val_X)


confusion_matrix = pd.crosstab(val_y, df_preds, rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()

print(classification_report(val_y, df_preds))


# # Summary of Logistic Regression Model

# In[179]:


log_model_stats = sm.GLM.from_formula("DEATH_EVENT ~ age + AgeGroup_Cat + anaemia + creatinine_phosphokinase + diabetes + ejection_fraction + high_blood_pressure + platelets + serum_creatinine + serum_sodium + sex + smoking + time", family=sm.families.Binomial(), data=df)
result = log_model_stats.fit()
result.summary() 


# # Create Classifer Logistic Regression Model - Limited Attributes

# In[190]:


#Making the model
log_model_lim = LogisticRegression(solver='liblinear')
 
#Fitting the model
log_model_lim.fit(X_lim, y)

#Cross Validation - 10
k = 10
kf = KFold(n_splits=k, random_state=None)
result = cross_val_score(log_model_lim , X_lim, y, cv = kf)
 
print("Avg accuracy: {}".format(result.mean()))


# In[187]:


#Confusion Matrix

#Create Test-Train Split
train_X, val_X, train_y, val_y = train_test_split(X_lim, y,random_state = 0)

#Create Model
log_model_lim_cm = LogisticRegression(solver= 'liblinear')

log_model_lim_cm.fit(train_X, train_y)

#Predict
df_preds = log_model_lim_cm.predict(val_X)


confusion_matrix = pd.crosstab(val_y, df_preds, rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()


# In[121]:


log_model_lim_stats = sm.GLM.from_formula("DEATH_EVENT ~ age + AgeGroup_Cat + anaemia + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium + time", family=sm.families.Binomial(), data=df)
result = log_model_lim_stats.fit()
result.summary() 


# In[ ]:


'''
Summary of Model Performance
    Attributes = all/limited
    Model = Logistic Regression

    The k-fold model showed 81.3% accuracy
    With limited attriutes, it only improved slightly to 83.0%
    
    Most significant features in dataset based off p values: ejection_fraction, serum_creatine
    Least significant: smoking, anaemia, high blood pressure
'''

