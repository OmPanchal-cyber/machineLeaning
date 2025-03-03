
import pandas as pd
data=pd.read_excel(r"C:\Users\panchal_____om\Downloads\home predication.xlsx")
data

data.drop(["Address","Date","Postcode","YearBuilt","Lattitude","Longtitude"],axis=1,inplace=True)
data

data[data["Propertycount"].isnull()].index

data.loc[18523,"Propertycount"]=0
data.loc[26888,"Propertycount"]=0
data.loc[29483,"Propertycount"]=0

data[data["Distance"].isnull()].index
data.loc[29483,"Distance"]=0
data[data["Bedroom2"].isnull()].inde
data.Bedroom2.fillna(data.Bedroom2.mean(),inplace=True)
data[data["Bathroom"].isnull()].index
data.Bathroom.fillna(data.Bathroom.mean(),inplace=True)
data[data["Car"].isnull()].index

data.Car.fillna(data.Car.mean(),inplace=True)
data.Landsize.fillna(data.Landsize.mean(),inplace=True)

data.BuildingArea.fillna(data.BuildingArea.min(),inplace=True)
data.Price.fillna(data.Price.min(),inplace=True)
data

data.info()
data.select_dtypes(include="object").columns
data["Type"]=data["Type"].replace(['h', 'u', 't'],[0,1,2],regex=True).astype(int)
data

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()"Suburb", "Method", "SellerG", "CouncilArea", "Regionname"

data[["Suburb", "Method", "SellerG", "CouncilArea", "Regionname"]]=data[["Suburb", "Method", "SellerG", "CouncilArea", "Regionname"]].apply(enc.fit_transform)
data

dum=pd.get_dummies(data["Suburb"]).astype(int)
dum

x=data.drop("Price",axis=1)
y=data["Price"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.4,random_state=20)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
model.score(x_test,y_test)

from sklearn.model_selection import KFold,cross_val_score
kf=KFold(n_splits=4)
cross_val_score(LinearRegression(),x,y,cv=kf).mean()

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
lis=[RandomForestRegressor,DecisionTreeRegressor,KNeighborsRegressor,SVR]

for i in lis:
    kf=KFold(n_splits=4)
    print(i)
    print(cross_val_score(i(),x,y,cv=kf).mean())
    
dic={"n_estimators":[100,200,300,400],"random_state":[10,20,30]}
model.fit(x,y)
from sklearn.model_selection import RandomizedSearchCV
random=RandomizedSearchCV(RandomForestRegressor(),param_distributions=dic,cv=kf,n_iter=20,verbose=2)
random.fit(x,y)
random.best_params_


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(random_state=20,n_estimators=400)
model.fit(x_train,y_train)
model.score(x_train,y_train)
model.score(x_test,y_test)

pd.Series(model.feature_importances_,index=x.columns).nlargest(15).plot(kind="barh")

