#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Mahipalsing Assignments


# In[ ]:


# 1) Prepare a classification model using Naive Bayes 
# for salary data 

# Data Description:

# age -- age of a person
# workclass	-- A work class is a grouping of work 
# education	-- Education of an individuals	
# maritalstatus -- Marital status of an individulas	
# occupation	 -- occupation of an individuals
# relationship -- 	
# race --  Race of an Individual
# sex --  Gender of an Individual
# capitalgain --  profit received from the sale of an investment	
# capitalloss	-- A decrease in the value of a capital asset
# hoursperweek -- number of hours work per week	
# native -- Native of an individual
# Salary -- salary of an individual


# In[3]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Load data sets


# In[5]:


salary_train = pd.read_csv(r"C:\Users\anupa\Downloads\SalaryData_Train.csv")
salary_train


# In[6]:


salary_test = pd.read_csv(r"C:\Users\anupa\Downloads\SalaryData_Test.csv")
salary_test


# In[7]:


salary_train.columns


# In[8]:


salary_test.columns


# In[9]:


salary_test.dtypes


# In[10]:


salary_train.dtypes


# In[11]:


salary_train.info()


# In[12]:


salary_test.info()


# In[13]:


string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# In[14]:


#Graphical Visualization


# In[15]:


sns.pairplot(salary_train)


# In[16]:


sns.pairplot(salary_test)


# In[17]:


sns.boxplot(salary_train['Salary'], salary_train['capitalgain'])


# In[18]:


sns.boxplot(salary_test['Salary'], salary_test['capitalgain'])


# In[19]:


sns.countplot(salary_train['Salary'])


# In[20]:


sns.countplot(salary_test['Salary'])


# In[21]:


plt.figure(figsize=(20,10))
sns.barplot(x='Salary', y='hoursperweek', data=salary_train)
plt.show()


# In[22]:


plt.figure(figsize=(20,10))
sns.barplot(x='Salary', y='hoursperweek', data=salary_test)
plt.show()


# In[23]:


plt.figure(figsize=(15,10))
sns.lmplot(y='capitalgain', x='hoursperweek',data=salary_train)
plt.show()


# In[24]:


plt.figure(figsize=(15,10))
sns.lmplot(y='capitalgain', x='hoursperweek',data=salary_test)
plt.show()


# In[25]:


plt.subplots(1,2, figsize=(16,8))

colors = ["#FF0000", "#64FE2E"]
labels ="capitalgain", "capitalloss"

plt.suptitle('salary of an individual', fontsize=18)

salary_train["Salary"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', shadow=True, colors=colors, 
                                             labels=labels, fontsize=10, startangle=25)


# In[26]:


plt.style.use('seaborn-whitegrid')

salary_train.hist(bins=20, figsize=(15,10), color='red')
plt.show()


# In[27]:


plt.subplots(1,2, figsize=(16,8))

colors = ["#FF0000", "#64FE2E"]
labels ="capitalgain", "capitalloss"

plt.suptitle('salary of an individual', fontsize=18)

salary_test["Salary"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', shadow=True, colors=colors, 
                                             labels=labels, fontsize=10, startangle=25)


# In[28]:


plt.style.use('seaborn-whitegrid')

salary_test.hist(bins=20, figsize=(15,10), color='red')
plt.show()


# In[29]:


#Preprocessing


# In[30]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()


# In[31]:


for i in string_columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])


# In[32]:


col_names=list(salary_train.columns)
col_names


# In[33]:


train_X=salary_train[col_names[0:13]]
train_X


# In[34]:


train_Y=salary_train[col_names[13]]
train_Y


# In[35]:


test_x=salary_test[col_names[0:13]]
test_x


# In[36]:


test_y=salary_test[col_names[13]]
test_y


# In[37]:


#Build Naive Bayes Model


# In[38]:


#Gaussian Naive Bayes


# In[39]:


from sklearn.naive_bayes import GaussianNB
Gnbmodel=GaussianNB()


# In[40]:


train_pred_gau=Gnbmodel.fit(train_X,train_Y).predict(train_X)
train_pred_gau


# In[41]:


test_pred_gau=Gnbmodel.fit(train_X,train_Y).predict(test_x)
test_pred_gau


# In[42]:


train_acc_gau=np.mean(train_pred_gau==train_Y)


# In[43]:


test_acc_gau=np.mean(test_pred_gau==test_y)


# In[44]:


train_acc_gau


# In[45]:


test_acc_gau


# In[46]:


#Multinomial Naive Bayes


# In[47]:


from sklearn.naive_bayes import MultinomialNB
Mnbmodel=MultinomialNB()


# In[48]:


train_pred_multi=Mnbmodel.fit(train_X,train_Y).predict(train_X)
train_pred_multi


# In[49]:


test_pred_multi=Mnbmodel.fit(train_X,train_Y).predict(test_x)
test_pred_multi


# In[50]:


train_acc_multi=np.mean(train_pred_multi==train_Y)
train_acc_multi


# In[51]:


test_acc_multi=np.mean(test_pred_multi==test_y)
test_acc_multi


# In[ ]:




