import pandas as pd

data=pd.read_csv(r"C:\Users\panchal_____om\Downloads\Naive Bayes Loan.csv")
data
data.isnull().sum()
data["BILL_AMT4"].mean()

for col in data.columns:
   # Calculate the mean of the non-null values in the column
   mean_val = data[col].mean(skipna=True)
   for i in range(len(data)):
       
       if pd.isnull(data.loc[i, col]):
           # Fill the null value with the mean
           data.loc[i, col] = mean_val
          
import matplotlib.pyplot as plt
plt.boxplot(data["AGE"])

data.select_dtypes(include="object").columns
data.info()
data["Default Status"].unique()

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
data[["Default Status"]]=data[["Default Status"]].apply(enc.fit_transform)
data.info()

x=data.drop("AGE",axis=1)
y=data["AGE"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=10)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_test,y_test)
model.fit(x_train,y_train)
model.score(x_train,y_train)
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
    print(cross_val_score(LinearRegression(),x,y,cv=kf).mean)
dic={"n_estimators":[100,200,300],"random_state":[10,20,30,40]}
model.fit(x,y)

from sklearn.model_selection import RandomizedSearchCV
random=RandomizedSearchCV(RandomForestRegressor(),param_distributions=dic,cv=kf,n_iter=20,verbose=2)
random.fit(x,y)
random.best_params_

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(random_state=20, n_estimators=300)

model.fit(x_train,y_train)
model.fit(x_train,y_train)

model.score(x_train,y_train)
model.score(x_test,y_test)

