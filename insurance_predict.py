#!/usr/bin/env python
# coding: utf-8

# In[137]:


mport pandas as pd


# In[138]:


data=pd.read_csv(r"C:\Users\panchal_____om\Downloads\Insurance Prediction.csv")


# In[139]:


data


# In[140]:


data["bmi"].isnull().sum()


# In[141]:


data["smoker"]=data["smoker"].replace(["yes","no"],[0,1],regex=True).astype(int)


# In[142]:


data["region"].unique()


# In[143]:


data["region"]=data["region"].replace(['southwest', 'southeast', 'northwest', 'northeast'],[0,1,2,3],regex=True).astype(int)


# In[144]:


data["sex"]=data["sex"].replace(["female","male"],[0,1],regex=True).astype(int)


# In[ ]:





# In[145]:


import matplotlib.pyplot as plt


# In[146]:


plt.scatter(data["age"],data["children"])


# In[147]:


plt.bar(data["bmi"],data["children"])
plt.xlabel("hello",color="yellow",fontsize="14")
plt.show()


# In[ ]:





# In[ ]:





# In[148]:


from sklearn.linear_model import LinearRegression


# In[149]:


model=LinearRegression()


# In[150]:


model.fit(data[["region"]],data["sex"])


# In[151]:


model.predict([[3]])


# In[152]:


data["predect"]=model.predict(data[["region"]])


# In[153]:


data


# In[ ]:





# In[ ]:





# In[ ]:





# In[154]:


plt.plot(data["region"],data["sex"])
plt.plot(data["region"],model.predict(data[["region"]]))


# In[155]:


from sklearn.model_selection import train_test_split


# In[156]:


train_test_split(x,y,train_size=0.2,random_state=20)


# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[ ]:


r2_score(data["sex"],model.predict(data[["region"]]))


# In[ ]:


mean_absolute_error(data["sex"],model.predict(data[["region"]]))


# In[ ]:


mean_squared_error(data["sex"],model.predict(data[["region"]]))


# In[ ]:


from sklearn.linear_model import Lasso,Ridge


# In[ ]:


model1=Lasso(alpha=2000000)
model1.fit(data[["region"]],data["sex"])


# In[ ]:


data["reeg"]=model1.predict(data[["region"]])


# In[ ]:


data


# In[ ]:


import numpy as np


# In[ ]:


model1.score(data[["region"]],data["sex"])


# In[ ]:


plt.plot(np.array(data["region"]),np.array(data["pridect"]))
plt.plot(np.array(data["region"]),np.array(data["reeg"]))

