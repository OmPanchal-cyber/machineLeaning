#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
data=pd.read_excel(r"C:\Users\panchal_____om\Downloads\home predication.xlsx")


# In[74]:


data


# In[77]:


data.drop(["Address","Date","Postcode","YearBuilt","Lattitude","Longtitude"],axis=1,inplace=True)


# In[79]:


data


# In[80]:


data[data["Propertycount"].isnull()].index


# In[81]:


data.loc[18523,"Propertycount"]=0
data.loc[26888,"Propertycount"]=0
data.loc[29483,"Propertycount"]=0


# In[82]:


data[data["Distance"].isnull()].index


# In[83]:


data.loc[29483,"Distance"]=0


# In[84]:


data[data["Bedroom2"].isnull()].index


# In[85]:


data.Bedroom2.fillna(data.Bedroom2.mean(),inplace=True)


# In[86]:


data[data["Bathroom"].isnull()].index


# In[87]:


data.Bathroom.fillna(data.Bathroom.mean(),inplace=True)


# In[88]:


data[data["Car"].isnull()].index


# In[89]:


data.Car.fillna(data.Car.mean(),inplace=True)


# In[90]:


data.Landsize.fillna(data.Landsize.mean(),inplace=True)


# In[118]:


data.BuildingArea.fillna(data.BuildingArea.min(),inplace=True)


# In[123]:


data.Price.fillna(data.Price.min(),inplace=True)


# In[120]:


data


# In[121]:


data.info()


# In[103]:


data.select_dtypes(include="object").columns


# In[106]:


data["Type"]=data["Type"].replace(['h', 'u', 't'],[0,1,2],regex=True).astype(int)


# In[107]:


data


# In[109]:


from sklearn.preprocessing import LabelEncoder


# In[110]:


enc=LabelEncoder()"Suburb", "Method", "SellerG", "CouncilArea", "Regionname"


# In[113]:


data[["Suburb", "Method", "SellerG", "CouncilArea", "Regionname"]]=data[["Suburb", "Method", "SellerG", "CouncilArea", "Regionname"]].apply(enc.fit_transform)


# In[130]:


data


# In[153]:


dum=pd.get_dummies(data["Suburb"]).astype(int)


# In[154]:


dum


# In[155]:


x=data.drop("Price",axis=1)


# In[156]:


y=data["Price"]


# In[157]:


from sklearn.model_selection import train_test_split


# In[158]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.4,random_state=20)


# In[159]:


from sklearn.linear_model import LinearRegression


# In[160]:


model=LinearRegression()


# In[161]:


model.fit(x,y)


# In[162]:


model.score(x_test,y_test)


# In[163]:


from sklearn.model_selection import KFold,cross_val_score


# In[164]:


kf=KFold(n_splits=4)


# In[165]:


cross_val_score(LinearRegression(),x,y,cv=kf).mean()


# In[166]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
lis=[RandomForestRegressor,DecisionTreeRegressor,KNeighborsRegressor,SVR]


# In[167]:


for i in lis:
    kf=KFold(n_splits=4)
    print(i)
    print(cross_val_score(i(),x,y,cv=kf).mean())


# In[168]:


dic={"n_estimators":[100,200,300,400],"random_state":[10,20,30]}


# In[169]:


model.fit(x,y)


# In[170]:


from sklearn.model_selection import RandomizedSearchCV


# In[171]:


random=RandomizedSearchCV(RandomForestRegressor(),param_distributions=dic,cv=kf,n_iter=20,verbose=2)


# In[172]:


random.fit(x,y)


# In[173]:


random.best_params_


# In[174]:


from sklearn.ensemble import RandomForestRegressor


# In[175]:


model=RandomForestRegressor(random_state=20,n_estimators=400)


# In[178]:


model.fit(x_train,y_train)


# In[180]:


model.score(x_train,y_train)


# In[181]:


model.score(x_test,y_test)


# In[182]:


pd.Series(model.feature_importances_,index=x.columns).nlargest(15).plot(kind="barh")

