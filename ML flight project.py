#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_excel(r"C:\Users\panchal_____om\Downloads\ML Live Flight Fare Resourses16963295320.xlsx")


# In[3]:


data


# In[4]:


data.isnull().sum()


# In[5]:


data.drop(data[data["Route"].isnull()].index,inplace=True)


# In[6]:


data


# In[7]:


data.isnull().sum()


# In[8]:


data.drop(["Route","Additional_Info"],axis=1,inplace=True)


# In[9]:


data


# In[10]:


data["journy_Month"]=pd.to_datetime(data["Date_of_Journey"]).dt.month
data["journy_day"]=pd.to_datetime(data["Date_of_Journey"]).dt.day


# In[11]:


data


# In[12]:


data["dep_min"]=pd.to_datetime(data["Dep_Time"]).dt.minute
data["dep_hour"]=pd.to_datetime(data["Dep_Time"]).dt.hour


# In[13]:


data


# In[14]:


data["arrival_min"]=pd.to_datetime(data["Arrival_Time"]).dt.minute
data["arrival_hour"]=pd.to_datetime(data["Arrival_Time"]).dt.hour


# In[15]:


data


# In[16]:


lis=data["Duration"]


# In[17]:


new_lis=[]
for i in lis:
    if len(i.split())==1:
        if "h" in i:
            i=i+" 0m"
        elif "m" in i:
            i="0h "+i
    new_lis.append(i)
    


# In[18]:


data["Duration"]=new_lis


# In[19]:


data


# In[20]:


data["dur_hr"]=data["Duration"].str.split().str[0].replace("h","",regex=True).astype(int)
data["dur_min"]=data["Duration"].str.split().str[1].replace("m","",regex=True).astype(int)


# In[21]:


data


# In[22]:


data["Total_Stops"]=data["Total_Stops"].replace(['non-stop', '2 stops', '1 stop', '3 stops','4 stops'],[0,2,1,3,4]).astype(int)


# In[23]:


data


# In[24]:


data.drop(["Duration"],axis=1,inplace=True)


# In[25]:


data


# In[26]:


data.drop(["Arrival_Time","Date_of_Journey","Dep_Time"],axis=1,inplace=True)


# In[27]:


data


# In[28]:


from sklearn.preprocessing import LabelEncoder


# In[29]:


enc=LabelEncoder()


# In[30]:


data[["Airline","Source","Destination"]]=data[["Airline","Source","Destination"]].apply(enc.fit_transform)


# In[31]:


data


# In[32]:


x=data.drop("Price",axis=1)


# In[33]:


y=data["Price"]


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.4,random_state=20)


# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


model=LinearRegression()


# In[38]:


model.fit(x,y)


# In[39]:


model.score(x_train,y_train)


# In[38]:


model.fit(x,y)


# In[40]:


model.score(x_test,y_test)


# In[41]:


from sklearn.model_selection import KFold,cross_val_score


# In[42]:


kf=KFold(n_splits=4)


# In[43]:


cross_val_score(LinearRegression(),x,y,cv=kf).mean()


# In[45]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from  sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
lis=[SVR,RandomForestRegressor,DecisionTreeRegressor,KNeighborsRegressor]


# In[46]:


for i in lis:
    kf=KFold(n_splits=4)
    print(i)
    print(cross_val_score(i(),x,y,cv=kf).mean())


# In[47]:


dic={"n_estimators":[100,200,300,400],"random_state":[10,20,30]}


# In[38]:


model.fit(x,y)


# In[48]:


from sklearn.model_selection import RandomizedSearchCV


# In[49]:


random=RandomizedSearchCV(RandomForestRegressor(),param_distributions=dic,cv=kf,n_iter=20,verbose=2)


# In[50]:


random.fit(x,y)


# In[51]:


random.best_params_


# In[52]:


from sklearn.ensemble import RandomForestRegressor


# In[54]:


model=RandomForestRegressor(random_state=20,n_estimators=400)


# In[56]:


model.fit(x_train,y_train)


# In[57]:


model.score(x_train,y_train)


# In[58]:


model.score(x_test,y_test)


# In[59]:


pd.Series(model.feature_importances_,index=x.columns).nlargest(15).plot(kind="barh")

