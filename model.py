#!/usr/bin/env python
# coding: utf-8

# #### Author : Sanjoy Biswas
# #### Topic : Linear Regression Tutorial With Project Solving
# #### Email : sanjoy.eee32@gmail.com

# Linear regression is one of the easiest and most popular Machine Learning algorithms. It is a statistical method that is used for predictive analysis. Linear regression makes predictions for continuous/real or numeric variables such as sales, salary, age, product price, etc.
# 
# Linear regression algorithm shows a linear relationship between a dependent (y) and one or more independent (y) variables, hence called as linear regression. Since linear regression shows the linear relationship, which means it finds how the value of the dependent variable is changing according to the value of the independent variable.

# #### Import Libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from word2number import w2n
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# #### Import Dataset

# In[21]:


df = pd.read_csv('hiring.csv')
df.head()


# #### Preprocessing Datasets

# In[3]:


df.experience = df.experience.fillna('Zero')


# In[4]:


df


# In[5]:


### Apply word_to_num
df.experience = df.experience.apply(w2n.word_to_num)


# In[6]:


df


# In[7]:


import math
median_test_score = math.floor(df['test_score(out of 10)'].mean())
median_test_score


# In[8]:


dff = df['test_score(out of 10)'].mean()
dff


# In[9]:


df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(dff)


# In[10]:


df


# In[11]:


### Show Columns Name
df.columns


# #### Features Selection

# In[12]:


predictors = ['experience', 'test_score(out of 10)', 'interview_score(out of 10)']
x = df[predictors]
y = df['salary($)']


# #### Split Train and test datasets

# In[13]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[14]:


x_train.shape,x_test.shape


# In[15]:


y_train.shape,y_test.shape


# #### Apply Linear Regression

# In[16]:


reg = LinearRegression()


# In[17]:


reg.fit(x_train,y_train)


# In[22]:


import pickle


# In[23]:


pickle.dump(reg,open('model.pkl','wb'))


# In[25]:


model = pickle.load(open('model.pkl','rb'))


# In[ ]:




