#!/usr/bin/env python
# coding: utf-8

# In[82]:


# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[83]:


# load the dataset

df=pd.read_csv('Iris.csv')


# In[84]:


df


# In[85]:


# Summarizing Dataset:

# The given dataset is about a particular species of flower (Iris-setosa,Iris-virginica & Iris-versicolor)
#Iris setosa, the bristle-pointed iris, is a species of flowering plant in the genus Iris of the family Iridaceae.
# The dataset shows the dimensional measurements of the various flower parts such as sepal the outer covering and petal the actaul part of flower.
# Each rows shows the actual dimensional measure of each flower as per the species type.
# Our objective is to predict which flower species, does the data belong to by follwing steps of handling dataset.


# In[86]:


df.info()


# In[87]:


df.describe()


# In[88]:


len(df)


# In[89]:


df.head(5)


# In[90]:


df.tail(5)


# In[91]:


df.dtypes


# In[92]:


df['Species'].unique()


# In[93]:


# From the above we can see that there are 150 entries.
# The dimensions of the features are calculated as maximum & mean along with "std" values.
# We can see that the target column has 3 major type of classification : 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
# Also, we can see that the dataset has only 1 column with 'object' data type which we will have to work upon.


# In[94]:


# Pre-processing the dataset


# In[95]:


# checking the null values

df.isnull().sum()


# In[96]:


df.isnull().sum().max


# In[97]:


# From the above it is clear that we do not have any null values in this dataset. so now we can proceed further.


# In[98]:


# We have a column "Id" with serial numbering for the entries in the dataset.

df['Id'].unique


# In[99]:


# We have to drop this column

df.drop('Id', axis=1, inplace=True)


# In[100]:


df


# In[101]:


# There is one column which is in "object" data types, so we will hve to perform label encoding in this dataset

df['Species'].unique()


# In[102]:


# We import additional libraries

from sklearn.preprocessing import LabelEncoder
print('importing is done')


# In[103]:


label = LabelEncoder()
label


# In[104]:


# Column to be encoded

columns_to_encode = ['Species']


# In[105]:


# Apply label encoding to each column to be encoded

df[columns_to_encode]=df[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[106]:


df


# In[107]:


df.dtypes


# In[108]:


df['Species'].unique()


# In[109]:


# The above is the final dataset with all integer values as data type and 'Species' column encoded for further processing.


# In[110]:


# Data Visualisation of the above dataset


# In[111]:


# A--- Scatter Plot

sns.pairplot(df, diag_kind='kde')


# In[112]:


# From the above chart we can see that there is prominent co-relation between sepal features with the type of species
# skewness is obsrved in the dataset for sepal and petal features.
# we can see that petal length is realted with petal width
# We can also see that there is minor corelation with sepal length and petal length.


# In[113]:


# B----- Correlation Heatmap

# Calculating correlation matrix for all the features.

corr = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[114]:


# The above chart givees us clear picture about direct co-relation of sepal feature as per species types.
# Petal features have minor corelation with sepal features.


# In[115]:


# C---- Bar Plots

# Grouping by 'Species' and calculating the mean for each feature
mean_values = df.groupby('Species').mean()

# Plotting the bar chart
mean_values.plot(kind='bar')
plt.title('Mean Feature Values by Species')
plt.xlabel('Species')
plt.ylabel('Mean Value')
plt.xticks(rotation=0)
plt.legend(title='Feature')
plt.show()


# In[116]:


# Defining : "0"--Iris-setosa, "1"---Iris-versicolor, "2"-----Iris-virginica.
# "2"-----Iris-virginica has the feature prominence as sepal length > Petal length > sepal width > petal width
# "1"---Iris-versicolor has the feature prominence as sepal length > petal length > sepal width > petal width
# "0"--Iris-setosa has the feature prominence as sepal length > sepal width > petal length > petal width.
# It can be concluded has Iris setosa has features which are manageable diffrentiate from the other 2 species
# For species type 1 and 2 the features are very similar making it difficult to identify them.


# In[117]:


# Segregating the Dataset into Input(x) and Output(y)

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[118]:


x


# In[119]:


y


# In[120]:


x.shape, y.shape


# In[121]:


# Splitting the Dataset into Training and Testing Data


# In[122]:


# we import additional libraries

from sklearn.model_selection import train_test_split
print('importing is done')


# In[123]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[124]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[125]:


# Loading the Models


# In[126]:


# We import additional files 

from xgboost import XGBClassifier
print ("importing is done")


# In[127]:


# Now we create a model for our dataset

xgbc = XGBClassifier()
xgbc


# In[128]:


# we fit the training data in our model

xgbc.fit(x_train,y_train)
xgbc


# In[129]:


# Now testing new Model prediction by providing Test Data set

y_pred = xgbc.predict(x_test)
y_pred


# In[130]:


y_test


# In[131]:


# Calculating the Accuracy of the Trained Models


# In[132]:


# we import additional library for accuracy

from sklearn.metrics import accuracy_score, classification_report
print('importing is done')


# In[133]:


# Now we check for accuracy of test data vs training data

accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))
np.round(accuracy,2)*100


# In[134]:


# Generate a detailed classification report

report = classification_report(y_test, y_pred)
print(report)


# In[135]:


# With XG Boost Classifier we get 97 % accuracy.


# In[136]:


# If we keep split as 70/30
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=42)
xgbc.fit(x_train, y_train)
y_pred = xgbc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
np.round(accuracy,2)*100


# In[137]:


# The accuracy remains the same even after changing the split to 70/30 ratio.


# In[138]:


# Predicting the Output of Single Test Data using the Trained Model

y_test


# In[139]:


x_test[2]


# In[140]:


x_test[2].shape


# In[141]:


x_test[2].reshape(1,4).shape


# In[142]:


xgbc.predict(x_test[2].reshape(1,4))


# In[143]:


y_test[2]


# In[144]:


# conclusion: Our model is well trained and provides an accurate prediction.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




