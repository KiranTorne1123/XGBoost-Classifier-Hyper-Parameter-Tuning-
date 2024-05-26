#!/usr/bin/env python
# coding: utf-8

# In[92]:


# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[93]:


# load the dataset

df=pd.read_csv('data.csv')


# In[94]:


df


# In[95]:


# Summarizing Dataset:

# This dataset is all about various patients readings during test done for predicting the type of Breast Cancer.
# Many Breast variables are taken into consideration such as radius, preimeter, area, concavity etc.
# This dataset has total of 569 patients readings summarizing to prediction of their respective Breast Cancer status.
# Our objective is to study these variables and predict about the affirmation of a particular type of cancer.


# In[96]:


df.info()


# In[97]:


df.describe().transpose()


# In[98]:


len(df)


# In[99]:


df.head(5).transpose()


# In[100]:


df.tail(5).transpose()


# In[101]:


df.dtypes


# In[102]:


# From the above we can cofirm that this dataset has total 569 patients details and 33 features under study as in columns.
# It shows that features which are involved to predict the cancer type accurately.
# Features from radius_worst till fractal_dimension_worst confirms about the cancer.
# Features from radius_mean till fractal_dimension_mean are derived after studies on a patient
# Features from radius_se till fractal_dimension_se are under observation for the prabable patient carrying cancer cells.
# Except 'diagnosis' column rest all are in numeric data types.
# We have "Unnamed: 32 " column to be filled with null values as "NaN".
# 'Id' column has details addressing to each patient by asigning them a particular number.


# In[103]:


# Pre-processing the dataset


# In[104]:


# Checking the null values

df.isnull().sum()


# In[105]:


df.isnull().sum().max


# In[106]:


# From the above we can see that the column'Unnamed: 32' has 569 null values.
# We will have to drop this entire column as it serves no purpose in this dataset.
df.drop('Unnamed: 32', axis=1, inplace=True)

# We have a column "Id" 
# We will drop the 'Id' column since it has serial numbering for the entries in the dataset & not required for our perspective in determining the cancer types.
df.drop('id', axis=1, inplace=True)


# In[107]:


df.info()


# In[108]:


# The below column is in "object" data types, so we will hve to perform label encoding in this dataset

df['diagnosis'].unique() # M == Malignant and B == Benign


# In[109]:


# We import additional libraries

from sklearn.preprocessing import LabelEncoder
print('importing is done')


# In[110]:


label = LabelEncoder()
label


# In[111]:


# Column to be encoded
columns_to_encode = ['diagnosis']


# In[112]:


# Apply label encoding to each column to be encoded

df[columns_to_encode]=df[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[113]:


df.head(5).transpose()


# In[114]:


df.dtypes


# In[115]:


df['diagnosis'].unique()


# In[116]:


df.isnull().sum().max


# In[117]:


# The above is our clean dataset.
# It has all data types as integers
# It does not have any null values
# Hence, we will be using this "df" for further analysis and processing.


# In[118]:


# Data Visualisation of the above dataset


# In[119]:


# Firstly we separate features into three types: mean, se, and worst

mean_features = df.filter(regex=".*_mean$")
se_features = df.filter(regex=".*_se$")
worst_features = df.filter(regex=".*_worst$")


# In[120]:


mean_features.transpose()


# In[121]:


se_features.transpose()


# In[122]:


worst_features.transpose()


# In[123]:


# A---SCATTER PLOT

# Now we can prepare scatter plots for each group against the diagnosis column
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plotting mean features
for feature in mean_features.columns:
    axs[0].scatter(df[feature], df['diagnosis'], alpha=0.5, label=feature)
axs[0].set_xlabel('Mean Features')
axs[0].set_ylabel('Diagnosis')
axs[0].set_title('Mean Features vs Diagnosis')
axs[0].legend()

# Plotting se features
for feature in se_features.columns:
    axs[1].scatter(df[feature], df['diagnosis'], alpha=0.5, label=feature)
axs[1].set_xlabel('SE Features')
axs[1].set_ylabel('Diagnosis')
axs[1].set_title('SE Features vs Diagnosis')
axs[1].legend()

# Plotting worst features
for feature in worst_features.columns:
    axs[2].scatter(df[feature], df['diagnosis'], alpha=0.5, label=feature)
axs[2].set_xlabel('Worst Features')
axs[2].set_ylabel('Diagnosis')
axs[2].set_title('Worst Features vs Diagnosis')
axs[2].legend()

plt.tight_layout()
plt.show()


# In[124]:


# B--- CORRELATION MATRIX
# Calculate correlation matrices
mean_corr_matrix = mean_features.corrwith(df['diagnosis'])
se_corr_matrix = se_features.corrwith(df['diagnosis'])
worst_corr_matrix = worst_features.corrwith(df['diagnosis'])

print("Correlation Matrix for Mean Features:")
print(mean_corr_matrix)

print("\nCorrelation Matrix for SE Features:")
print(se_corr_matrix)

print("\nCorrelation Matrix for Worst Features:")
print(worst_corr_matrix)


# In[125]:


# Plotting correlation matrix heatmap for mean features

plt.figure(figsize=(10, 6))
sns.heatmap(mean_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap for Mean Features vs Diagnosis')
plt.show()


# In[126]:


# The above heatmap shows most correlation of "mean" features with the diagnosis.


# In[127]:


# Plotting correlation matrix heatmap for SE features

plt.figure(figsize=(10, 6))
sns.heatmap(se_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap for SE Features vs Diagnosis')
plt.show()


# In[128]:


# The above heatmap shows mix correlation of "se" features with the diagnosis.


# In[129]:


# Plotting correlation matrix heatmap for worst features

plt.figure(figsize=(10, 6))
sns.heatmap(worst_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap for Worst Features vs Diagnosis')
plt.show()


# In[130]:


# The above heatmap shows moderate correlation of "worst" features with the diagnosis.


# In[131]:


# Conclusion: 
# The "mean" and "worst" heatmaps are more understandable than "se".
# The reason being there can be miscalculations or improper test performed on the patient.


# In[132]:


# Segregating the Dataset into Input(x) and Output(y)

x = df.drop(columns=['diagnosis']).values
y =  df['diagnosis'].values


# In[133]:


x


# In[134]:


y


# In[135]:


x.shape, y.shape


# In[136]:


# Splitting the Dataset into Training and Testing Data


# In[137]:


# we import additional libraries

from sklearn.model_selection import train_test_split
print('importing is done')


# In[138]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=17)


# In[139]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[140]:


# Loading the Models


# In[141]:


# We import additional files 

from xgboost import XGBClassifier
print ("importing is done")


# In[142]:


# Now we create a model for our dataset

xgbc = XGBClassifier()
xgbc


# In[143]:


# we fit the training data in our model

xgbc.fit(x_train,y_train)
xgbc


# In[144]:


# Now testing new Model prediction by providing Test Data set

y_pred = xgbc.predict(x_test)
y_pred


# In[145]:


y_test


# In[146]:


# Calculating the Accuracy of the Trained Models


# In[147]:


# we import additional library for accuracy

from sklearn.metrics import accuracy_score, classification_report
print('importing is done')


# In[148]:


# Now we check for accuracy of test data vs training data

accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))
np.round(accuracy,2)*100


# In[149]:


# Generate a detailed classification report

report = classification_report(y_test, y_pred)
print(report)


# In[150]:


# We get 97 % accuracy with 80/20 split.


# In[151]:


# If we keep split as 70/30
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=17)
xgbc.fit(x_train, y_train)
y_pred = xgbc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
np.round(accuracy,2)*100


# In[152]:


# The accuracy increases to 98 % after changing the split to 70/30 ratio.


# In[153]:


# Predicting the Output of Single Test Data using the Trained Model

y_test


# In[154]:


x_test[2]


# In[155]:


x_test[2].shape


# In[156]:


x_test[2].reshape(1,30).shape


# In[157]:


xgbc.predict(x_test[2].reshape(1,30))


# In[158]:


y_test[2]


# In[159]:


# conclusion: Our model is well trained and provides an accurate prediction.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




