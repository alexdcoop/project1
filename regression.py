#This is our project python file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read in the data
#DATA = pd.read_csv('https://raw.githubusercontent.com/alexdcoop/project1/main/fivesecbars.csv?token=AVII23MICGV4GPU7M72T3JTBLH4NG')
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
