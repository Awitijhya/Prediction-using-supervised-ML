#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation
# **Data Science & Business Analytics - June 2021**
# 
# **Task 1- Prediction using supervised ML**
# 
# 
# Problem Statement:
# **What will be predicted score if a student studies for 9.25 hrs/ day?**
# 
#    Submitted by:-Awitijhya Chakraborty
#    
#    Importing necessary libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


url="http://bit.ly/w-data"
data=pd.read_csv(url)
print('Data sucessfully loaded')


# In[4]:


data.head(10)


#  **Understanding the data**

# In[6]:


data.shape


# In[8]:


data.info()


# In[12]:


font1={'family':'calibri','color':'green','size':20}
font2={'family':'serif','color':'darkred','size':15}
data.plot(x='Hours',y='Scores',style='o',c='blue')
plt.title('Hours vs Score',fontdict=font1)
plt.xlabel('Hours studied',fontdict=font2)
plt.ylabel('Score obtained',fontdict=font2)
plt.show()


# In[13]:


data.corr()


# In[14]:


data.isnull().sum()


# In[15]:


x=(data['Hours'].values).reshape(-1,1)
y=data['Scores'].values


# In[16]:


x


# In[17]:


y


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print('splitting is done')


# In[19]:


from sklearn.linear_model import LinearRegression
regn = LinearRegression()
regn.fit(x_train,y_train)
print('tranning is done')


# In[20]:


print('Intercept value is :',regn.intercept_)
print('Linear coefficient is:',regn.coef_)


# In[24]:


# plotting the regression line
line = regn.coef_*x+regn.intercept_

#plotting for the the test data 
plt.scatter(x,y,c='blue')
plt.title('Linear Regression vs trained model',fontdict=font1)
plt.xlabel('Hours studied',fontdict=font2)
plt.ylabel('Score obtained',fontdict=font2)
plt.plot(x, line);
plt.show()


# In[25]:


#to predict scores of testing data
y_pred = regn.predict(x_test)


# In[26]:


y_pred


# In[28]:


df=pd.DataFrame({'Actual':y_test,'predicted':y_pred})


# In[29]:


df


# In[37]:


plt.scatter(y_test,y_pred,c='blue')
plt.show()


# In[ ]:


get_ipython().set_next_input('what will be predicted score if a student studies for 9.25 hrs/day');get_ipython().run_line_magic('pinfo', 'day')


# In[40]:


hours=9.25
pred_score=regn.predict([[hours]])
print("Number of hours={}".format(hours))
print("Predicted score ={}".format(pred_score[0]))


#  **evaluating the model**

# In[43]:


from sklearn import metrics
print('Mean Absolute Error:',
      metrics.mean_absolute_error(y_test,y_pred))


# **CONCLUSION
# for a student stuyding 9.25Hrsa day, the modl predicts his score as 93.6917**
