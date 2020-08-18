#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data=pd.read_csv("student_scores - student_scores.csv")
data


# In[17]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[37]:


X.reshape(1,-1)


# In[10]:


y


# In[38]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 


# In[5]:


model = LinearRegression()


# In[39]:


model.fit(X_train,y_train)


# In[40]:


print(model.coef_)
print(model.intercept_)


# In[41]:


# Plotting the regression line
line = model.coef_*X+model.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[42]:


y_pred=model.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[55]:


hour=[[9.5]]
new_pred = model.predict(hour)
df1=pd.DataFrame({'hour': hours, 'predicted_value': new_pred})
df1


# In[ ]:





# In[33]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 

