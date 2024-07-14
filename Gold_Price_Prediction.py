#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import steamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from PIL import Image



# In[4]:


gold = pd.read_csv('gld_price_data.csv')


# In[5]:


gold.head()


# In[6]:


gold.shape 


# In[7]:


gold.info()


# In[8]:


#check no. of missing values
gold.isnull().sum()


# In[12]:


#get statistical measure of data 
gold.describe()


# In[14]:


numeric_gold = gold.select_dtypes(include=[float, int])


# In[15]:


correlation = numeric_gold.corr()


# In[16]:


print(correlation)


# In[21]:


gold ['date_column'] = pd.to_datetime(gold ['Date'], errors='coerce')

# Select only numeric columns
numeric_gold= gold .select_dtypes(include=[float, int])

# Compute the correlation matrix
correlate = numeric_gold .corr()

print(correlate)


# In[23]:


plt.figure(figsize=(6,6))
sns.heatmap(correlation)


# In[29]:


plt.figure(figsize=(6,6))
sns.heatmap(correlation,cbar=True,square=True,annot=True,annot_kws={"size":8},cmap="plasma")


# In[36]:


sns.heatmap(correlation,annot=True,annot_kws={"size":8})


# In[37]:


print(correlation['GLD'])


# In[38]:


sns.displot(gold['GLD'],color='blue')


# In[39]:


X= gold.drop(['Date','GLD'],axis=1)
Y= gold['GLD']


# In[41]:


Y.shape


# In[53]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[58]:


print(X_train.dtypes)


# In[59]:


X_train = X_train.select_dtypes(include=[float, int])
X_test = X_test.select_dtypes(include=[float, int])


# In[60]:


reg = RandomForestRegressor()
reg.fit(X_train, Y_train)
pred = reg.predict(X_test)
score = r2_score(Y_test, pred)
print(f'R^2 Score: {score}')


# In[61]:


print(pred)


# In[ ]:
# web app
st.title('Gold Price Model')
img= Image.open('img3.jpeg')
st.image(img)
st.subheader('Using randomforestregressor')
st.write(score)                
                



