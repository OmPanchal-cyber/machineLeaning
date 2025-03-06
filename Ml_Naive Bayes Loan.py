#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv(r"C:\Users\panchal_____om\Downloads\Naive Bayes Loan.csv")


# In[9]:


data


# In[4]:


data.isnull().sum()


# In[5]:


data["BILL_AMT4"].mean()


# In[6]:


for col in data.columns:
   # Calculate the mean of the non-null values in the column
   mean_val = data[col].mean(skipna=True)
   for i in range(len(data)):
       
       if pd.isnull(data.loc[i, col]):
           # Fill the null value with the mean
           data.loc[i, col] = mean_val


# In[10]:


import matplotlib.pyplot as plt


# In[13]:


plt.boxplot(data["AGE"])


# In[14]:


data.select_dtypes(include="object").columns


# In[15]:


data.info()


# In[19]:


data["Default Status"].unique()


# In[20]:


from sklearn.preprocessing import LabelEncoder


# In[21]:


enc=LabelEncoder()


# In[22]:


data[["Default Status"]]=data[["Default Status"]].apply(enc.fit_transform)


# In[25]:


data.info()


# In[26]:


x=data.drop("AGE",axis=1)


# In[27]:


y=data["AGE"]


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=10)


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


model=LinearRegression()


# In[32]:


model.fit(x_test,y_test)
model.fit(x_train,y_train)


# In[33]:


model.score(x_train,y_train)


# In[34]:


model.score(x_test,y_test)


# In[35]:


from sklearn.model_selection import KFold,cross_val_score


# In[36]:


kf=KFold(n_splits=4)


# In[37]:


cross_val_score(LinearRegression(),x,y,cv=kf).mean()


# In[38]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
lis=[RandomForestRegressor,DecisionTreeRegressor,KNeighborsRegressor,SVR]


# In[39]:


for i in lis:
    kf=KFold(n_splits=4)
    print(i)
    print(cross_val_score(LinearRegression(),x,y,cv=kf).mean)


# In[40]:


dic={"n_estimators":[100,200,300],"random_state":[10,20,30,40]}


# In[41]:


model.fit(x,y)


# In[42]:


from sklearn.model_selection import RandomizedSearchCV


# In[43]:


random=RandomizedSearchCV(RandomForestRegressor(),param_distributions=dic,cv=kf,n_iter=20,verbose=2)


# In[44]:


random.fit(x,y)


# In[45]:


random.best_params_


# In[47]:


from sklearn.ensemble import RandomForestRegressor


# In[51]:


model=RandomForestRegressor(random_state=20, n_estimators=300)


# In[60]:


model.fit(x_train,y_train)
model.fit(x_train,y_train)


# In[59]:


model.score(x_train,y_train)


# In[62]:


model.score(x_test,y_test)

