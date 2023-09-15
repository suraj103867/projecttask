#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading libraries and the datasets
import pandas as pd 
import numpy as np
import seaborn as sms
import matplotlib.pyplot as plt


# In[2]:


da=pd.read_csv('advertising.csv')


# In[3]:


da.head()


# In[4]:


da.info()


# In[5]:


da.duplicated().sum()


# In[6]:


da.describe()


# In[7]:


da.columns


# In[8]:


da.shape


# In[9]:


y=da.drop('Newspaper',axis=1)


# In[10]:


y.shape


# In[11]:


x=da['Sales']


# In[12]:


x.shape


# In[13]:


sms.pairplot(da,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')
plt.show()


# In[ ]:


# Here if we analysis regarding the sales then Tv is quite predicatble as it is increasing.
# But we can't predict regarding the sales of the Radio and Newspapers.


# In[14]:


da['Newspaper'].plot.hist(bins=10,color='purple',xlabel='Newspaper')


# In[15]:


da['Radio'].plot.hist(bins=10,color='green',xlabel='Radio')


# In[16]:


da['TV'].plot.hist(bins=10)


# In[17]:


sms.heatmap(da.corr(),annot=True)
plt.show()


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1)


# In[20]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[21]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(da[['TV']],da[['Sales']],test_size=0.3,random_state=0)


# In[23]:


print(x_train)


# In[24]:


print(y_train)


# In[25]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[26]:


res=model.predict(x_test)
print(res)


# In[27]:


model.coef_


# In[28]:


model.intercept_


# In[29]:


plt.plot(res)


# In[30]:


plt.scatter(x_test,y_test)
plt.plot(x_test,7.14382225 + 0.05473199 * x_test,'r')
plt.show()


# In[ ]:


# concluding with saying that we have predicted the sales of the adversiting dataset.

