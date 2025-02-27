
import pandas as pd
data=pd.read_excel(r"C:\Users\panchal_____om\Downloads\ML Live Flight Fare Resourses16963295320.xlsx")
data


data.isnull().sum()
data.drop(data[data["Route"].isnull()].index,inplace=True)

data
data.isnull().sum()

data.drop(["Route","Additional_Info"],axis=1,inplace=True)
data

data["journy_Month"]=pd.to_datetime(data["Date_of_Journey"]).dt.month
data["journy_day"]=pd.to_datetime(data["Date_of_Journey"]).dt.day
data

data["dep_min"]=pd.to_datetime(data["Dep_Time"]).dt.minute
data["dep_hour"]=pd.to_datetime(data["Dep_Time"]).dt.hour
data

data["arrival_min"]=pd.to_datetime(data["Arrival_Time"]).dt.minute
data["arrival_hour"]=pd.to_datetime(data["Arrival_Time"]).dt.hour
data

lis=data["Duration"]
new_lis=[]
for i in lis:
    if len(i.split())==1:
        if "h" in i:
            i=i+" 0m"
        elif "m" in i:
            i="0h "+i
    new_lis.append(i)    

data["Duration"]=new_lis

data

data["dur_hr"]=data["Duration"].str.split().str[0].replace("h","",regex=True).astype(int)
data["dur_min"]=data["Duration"].str.split().str[1].replace("m","",regex=True).astype(int)

data
data["Total_Stops"]=data["Total_Stops"].replace(['non-stop', '2 stops', '1 stop', '3 stops','4 stops'],[0,2,1,3,4]).astype(int)

data

data.drop(["Duration"],axis=1,inplace=True)
data

data.drop(["Arrival_Time","Date_of_Journey","Dep_Time"],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()

data[["Airline","Source","Destination"]]=data[["Airline","Source","Destination"]].apply(enc.fit_transform)
data

x=data.drop("Price",axis=1)
y=data["Price"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.4,random_state=20)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
model.score(x_train,y_train)
model.fit(x,y)
model.score(x_test,y_test)
from sklearn.model_selection import KFold,cross_val_score
kf=KFold(n_splits=4)
cross_val_score(LinearRegression(),x,y,cv=kf).mean()

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from  sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
lis=[SVR,RandomForestRegressor,DecisionTreeRegressor,KNeighborsRegressor]

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

