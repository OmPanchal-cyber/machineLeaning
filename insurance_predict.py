
import pandas as pd
data=pd.read_csv(r"C:\Users\panchal_____om\Downloads\Insurance Prediction.csv")
data

data["bmi"].isnull().sum()
data["smoker"]=data["smoker"].replace(["yes","no"],[0,1],regex=True).astype(int)
data["region"].unique()
data["region"]=data["region"].replace(['southwest', 'southeast', 'northwest', 'northeast'],[0,1,2,3],regex=True).astype(int)
data["sex"]=data["sex"].replace(["female","male"],[0,1],regex=True).astype(int)



import matplotlib.pyplot as plt
plt.scatter(data["age"],data["children"])

plt.bar(data["bmi"],data["children"])
plt.xlabel("hello",color="yellow",fontsize="14")
plt.show()

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(data[["region"]],data["sex"])
model.predict([[3]])
data["predect"]=model.predict(data[["region"]])




data

plt.plot(data["region"],data["sex"])
plt.plot(data["region"],model.predict(data[["region"]]))

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
r2_score(data["sex"],model.predict(data[["region"]]))
mean_absolute_error(data["sex"],model.predict(data[["region"]]))
mean_squared_error(data["sex"],model.predict(data[["region"]]))



from sklearn.linear_model import Lasso,Ridge
model1=Lasso(alpha=2000000)
model1.fit(data[["region"]],data["sex"])
data["reeg"]=model1.predict(data[["region"]])

data


import numpy as np

model1.score(data[["region"]],data["sex"])
plt.plot(np.array(data["region"]),np.array(data["pridect"]))
plt.plot(np.array(data["region"]),np.array(data["reeg"]))

