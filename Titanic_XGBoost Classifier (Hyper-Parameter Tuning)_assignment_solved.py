#!/usr/bin/env python
# coding: utf-8

# In[219]:


# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[220]:


# load the dataset

df1=pd.read_csv('gender_submission.csv')
df2=pd.read_csv('train.csv')
df3=pd.read_csv('test.csv')
print('loading is done')


# In[221]:


df1


# In[ ]:


# Summarizing the dataset.
# Dataset "df1" shows only the passenger id along with their survival/non survival entries.
# Survival as "1" and Non-Survival is "0"


# In[222]:


df2


# In[ ]:


# Summarizing the dataset.
# Name --- Passenger name
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex(male/female)
# Age	Age in years
# sibsp	# of siblings / spouses aboard the Titanic	
# parch	# of parents / children aboard the Titanic	
# ticket	Ticket number
# fare	Passenger fare
# cabin	Cabin number
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton


# In[223]:


df3


# In[ ]:


# Summarizing the dataset.
# Dataset "df3" again shows all the passenger details except survival data in it.


# In[224]:


# Pre-processing the dataset


# In[225]:


# Before we proceed further we will have to merge the 2 datasets: "df1" & "df3"
# Merge the two datasets based on 'PassengerId'

df4= pd.merge(df1, df3, on='PassengerId', how='inner')


# In[226]:


df4


# In[227]:


# Merge the two datasets 'df2' and 'df4', again to form one single dataset to allow us to proceed further.

df = pd.concat([df2, df4])
df


# In[228]:


df.info()


# In[229]:


df.describe()


# In[230]:


len(df)


# In[231]:


df.head(5)


# In[232]:


df.tail(5)


# In[233]:


df.dtypes


# In[234]:


# The bove dataset has total of 1309 rows and 12 columns.
# Here we have 5 columns with data as object type. we will check if there is any null values and then proceed.


# In[235]:


# Now we check for null values

df.isnull().sum()


# In[236]:


# we have found 263 entries in Age column, 2 entries in Embarked column and 1014 entries in Cabin column as null values.
# we drop both the columns with null values.

df.dropna(axis=1, inplace=True)


# In[237]:


df


# In[238]:


# now we will work upon changing the columns from object data types to integer
# we can see column Name, sex and ticket is in object type so we will use label encoding on it.


# In[239]:


# We import additional libraries

from sklearn.preprocessing import LabelEncoder
print('importing is done')


# In[240]:


label = LabelEncoder()
label


# In[241]:


# Column to be encoded
columns_to_encode = ['Name','Sex','Ticket']


# In[242]:


# Apply label encoding to each column to be encoded

df[columns_to_encode]=df[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[243]:


#This is the final dataset with all integer values

df


# In[244]:


df.info()


# In[245]:


df.isnull().sum().max


# In[246]:


# From the above all codes we were able to transform df dataset into integer types which does not have any null values.
# The above dataset we will use as our final form to process further.


# In[247]:


# Data Visualisation of the above dataset


# In[248]:


# A----BAR PLOTS


survival_counts = df['Survived'].value_counts()

plt.bar(survival_counts.index, survival_counts.values)

plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Number of Survivors')

for i, count in enumerate(survival_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')
    
plt.show()


# In[249]:


# So the total count happens to be as follow
total_passengers_onboard = (815+494)
print("Total_passengers_onboard:", total_passengers_onboard)
total_death = 494
print("Total_Death:", total_death)
total_survived = 815
print("Total_Survived:", total_survived)


# In[250]:


# From the above chart we can conclude that no.of deaths is 494 and no.of survived people is 815.
# Total no.of passengers onboard were 1309 ( as per this dataset)


# In[251]:


# B---- Calculate correlation matrix for all the features(test_data)

corr = df.corr()
plt.figure(figsize=(20,12))
plt.title('Heatmap showing Correlation between all the features', fontsize=20)
sns.heatmap(corr, annot=True, cmap='mako', fmt=".2f")


# In[252]:


# The above heat map shows data with more survival rate than death rate.
# FROM THE ABOVE WE CAN SEE THAT THE BAR PLOTS GIVES US BETTER RESULT.


# In[253]:


# Segregating the Dataset into Input(x) and Output(y)

x = df.drop(columns=['Survived']).values
y = df['Survived'].values


# In[254]:


x


# In[255]:


y


# In[256]:


x.shape, y.shape


# In[257]:


# Splitting the Dataset into Training and Testing Data


# In[258]:


# we import additional libraries


from sklearn.model_selection import train_test_split
print('importing is done')


# In[324]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=6)


# In[325]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[326]:


# Loading the Models


# In[327]:


# We import additional files 

from xgboost import XGBClassifier
print ("importing is done")


# In[328]:


# Now we create a model for our dataset

xgbc = XGBClassifier()
xgbc


# In[329]:


# we fit the training data in our model

xgbc.fit(x_train,y_train)
xgbc


# In[330]:


# Now testing new Model prediction by providing Test Data set

y_pred = xgbc.predict(x_test)
y_pred


# In[331]:


y_test


# In[332]:


# Calculating the Accuracy of the Trained Models


# In[333]:


# we import additional library for accuracy

from sklearn.metrics import accuracy_score, classification_report
print('importing is done')


# In[334]:


# Now we check for accuracy of test data vs training data

accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))
np.round(accuracy,2)*100


# In[335]:


# Generate a detailed classification report

report = classification_report(y_test, y_pred)
print(report)


# In[336]:


# With XG Boost Classifier we get 87 % accuracy.


# In[339]:


# If we keep split as 70/30
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=6)
xgbc.fit(x_train, y_train)
y_pred = xgbc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
np.round(accuracy,2)*100


# In[340]:


# The accuracy remains the same even after changing the split to 70/30 ratio.


# In[341]:


# Predicting the Output of Single Test Data using the Trained Model

y_test


# In[348]:


x_test[7]


# In[349]:


x_test[7].shape


# In[350]:


x_test[7].reshape(1,7).shape


# In[351]:


xgbc.predict(x_test[7].reshape(1,7))


# In[353]:


y_test[7]


# In[354]:


# conclusion: Even though our model is at 87 % accuracy, it does provide an accurate prediction.


# In[355]:


# Now we try using Grad Boost Classifier on this same dataset


# In[356]:


# We import additional libraries

from sklearn.ensemble import GradientBoostingClassifier
print('importing is done')


# In[357]:


# Now we create a model for our dataset

gbc = GradientBoostingClassifier()
gbc


# In[358]:


# we fit the training data in our model

gbc.fit(x_train,y_train)
gbc


# In[359]:


# Now testing new Model prediction by providing Test Data set

y_pred = gbc.predict(x_test)
y_pred


# In[360]:


y_test


# In[361]:


# Calculating the Accuracy of the Trained Models

accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy : {0:0.3f}'. format (accuracy))


# In[362]:


np.round(accuracy,2)*100


# In[363]:


# Generate a detailed classification report

report = classification_report(y_test, y_pred)
print(report)


# In[364]:


# With Gradient Boosting Classifier we get 89 % accuracy.


# In[ ]:


# Conclusion: Grad boost classifier is better for this dataset as compared to XG Boost classifier.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




