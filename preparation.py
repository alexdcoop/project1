#This is our project python file
import pandas as pd
from pandas.core.accessor import CachedAccessor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read in the data
DATA = pd.read_csv('fivesecbars.csv')

#Adding a column converting epoch time to date time

DATA['time'] = pd.to_datetime(DATA['epochtime'],unit='s')



#Making a seperate dataframe per ticker ID
TICKER0= DATA.loc[DATA['tickerid']==0]
TICKER1= DATA.loc[DATA['tickerid']==1]
TICKER2= DATA.loc[DATA['tickerid']==2]
TICKER3= DATA.loc[DATA['tickerid']==3]
TICKER4= DATA.loc[DATA['tickerid']==4]

#Plotting The opening epoch price by time in groups. 

#This data is over a week
sns.lineplot(data=TICKER0, x="time", y="open")
plt.xticks(rotation=45)
plt.title('Ticker0')
plt.ylabel('Price')
plt.show()

#The rest of these plots are over 1 day
plt.subplot(1,2,1)
sns.lineplot(data=TICKER1, x="time", y="open")
plt.xticks(rotation=45)
plt.title('Ticker1')
plt.ylabel('Price')


plt.subplot(1,2,2)
sns.lineplot(data=TICKER2, x="time", y="open")
plt.xticks(rotation=45)
plt.title('Ticker2')
plt.ylabel('Price')
plt.show()

plt.subplot(1,2,1)
sns.lineplot(data=TICKER3, x="time", y="open")
plt.xticks(rotation=45)
plt.title('Ticker3')
plt.ylabel('Price')

plt.subplot(1,2,2)
sns.lineplot(data=TICKER4, x="time", y="open")
plt.xticks(rotation=45)
plt.title('Ticker4')
plt.ylabel('Price')
plt.show() 

# Support vector machine - Alex Cooper
from sklearn import svm

#Create X features
feature_cols = ["open","high","low","close","volume"]
X = DATA.loc[:,feature_cols]

#Create response vector
Y = DATA.weightedavgprice

#Build SVR model
regr = svm.SVR(cache_size=1000)
regr.fit(X,Y)

#Predict values
predictedValues = regr.predict(X.head(1000))

####add some visualization from predicted values below

#plot predicted values vs. actual values

#KEVIN's work
from sklearn import linear_model, preprocessing
import sklearn as sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

le= preprocessing.LabelEncoder()
weightedprice=le.fit_transform(list(DATA['weightedavgprice']))
time=le.fit_transform(list(DATA['time']))
volume=le.fit_transform(list(DATA['volume']))
open=le.fit_transform(list(DATA['open']))
low=le.fit_transform(list(DATA['low']))
high=le.fit_transform(list(DATA['high']))
close=le.fit_transform(list(DATA['close']))
epoch=le.fit_transform(list(DATA['epochtime']))
tickerid=le.fit_transform(list(DATA['tickerid']))

#alex CV
from sklearn.model_selection import GridSearchCV

knn2 = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1,25)}

knn_gscv = GridSearchCV(knn2, param_grid=param_grid, cv=5)

#fit knn
knn_gscv.fit(x_train,y_train)

#check accuracy
knn_gscv.score(x_test,y_test) 

#top performing n_neighbors value
knn_gscv.best_params_


X= list(zip(epoch, volume, open, low, high,tickerid))
Y= list(close)

x_train, x_test, y_train,y_test=sklearn.model_selection.train_test_split(X,Y, test_size=0.2, random_state = 0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train) #fit model to training set

acc=knn.score(x_test,y_test) 
print(acc)  #accuracy on training data

y_pred = knn.predict(x_test)

y_pred

print("Model accuracy score: {0:0.4f}".format(accuracy_score(y_test, y_pred)) )   #Model test accuracy 

#Cross validation
knn_cv = KNeighborsClassifier(n_neighbors=3)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, Y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores)


results=pd.DataFrame(x_test)
results['pred']=y_pred
results.rename(columns={0: "epoch", 1: "volume",2:'open',3:'low',4:'high',5:'tickerid'})
results1=results.loc[results[5] == 0]
results1=results1.sort_values(by=0,ascending=True)
sns.lineplot(data=results1, x=0, y="pred")
plt.xticks(rotation=45)
plt.title('time')
plt.ylabel('Price')
plt.show()

results2=results.loc[results[5] == 1]
results2=results2.sort_values(by=0,ascending=True)
sns.lineplot(data=results2, x=0, y="pred")
plt.xticks(rotation=45)
plt.title('time')
plt.ylabel('Price')
plt.show()

results3=results.loc[results[5] == 2]
results3=results3.sort_values(by=0,ascending=True)
sns.lineplot(data=results3, x=0, y="pred")
plt.xticks(rotation=45)
plt.title('time')
plt.ylabel('Price')
plt.show()

results4=results.loc[results[5] == 3]
results4=results4.sort_values(by=0,ascending=True)
sns.lineplot(data=results4, x=0, y="pred")
plt.xticks(rotation=45)
plt.title('time')
plt.ylabel('Price')
plt.show()

results5=results.loc[results[5] == 4]
results5=results5.sort_values(by=0,ascending=True)
sns.lineplot(data=results5, x=0, y="pred")
plt.xticks(rotation=45)
plt.title('time')
plt.ylabel('Price')
plt.show()




#regression - Noah
TICKER0
from sklearn import linear_model, preprocessing

#original regression
x = TICKER0['time']
y = TICKER0['weightedavgprice']

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

#kevin regression
x = DATA.iloc[:,1].values.reshape(-1,1)
y = DATA.iloc[:,7].values.reshape(-1,1)

linear_regressor = LinearRegression()
linear_regressor(x,y)
y.pred = linear_regressor(x)
plt.scatter(x,y)
plt.show(x,y.pred, color = 'red')
plt.show()

#regression -- predicting for ticker 0 
#predicting 10 values into the future
print(DATA)
print(TICKER0)

newticker0 = TICKER0[['weightedavgprice']]

import pandas_ta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

newticker0.ta.ema(close = 'weighted_avgprice',length = 10, append = True)
print(newticker0.head(10))

newticker0 = newticker0.iloc[10:]
print(newticker0.head(10))

X_train, X_test, y_train, y_test = train_test_split(newticker0[['weightedavgprice']], newticker0[['EMA_10']],test_size=.2)
print(X_test.describe())
print(X_train.describe())

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


print("Model Coefficients:", model.coef_)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination:", r2_score(y_test, y_pred))
