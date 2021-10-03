#This is our project python file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Read in the data
DATA = pd.read_csv('https://raw.githubusercontent.com/alexdcoop/project1/main/fivesecbars.csv?token=AVII23MICGV4GPU7M72T3JTBLH4NG')

#Making a seperate dataframe per ticker ID
TICKER0= DATA.loc[DATA['tickerid']==0]
TICKER1= DATA.loc[DATA['tickerid']==1]
TICKER2= DATA.loc[DATA['tickerid']==2]
TICKER3= DATA.loc[DATA['tickerid']==3]
TICKER4= DATA.loc[DATA['tickerid']==4]

#Plotting The opening epoch price by time in groups. 
plt.subplot(1,2,1)
sns.lineplot(data=TICKER0, x="epochtime", y="open")
plt.title('Ticker0')

plt.subplot(1,2,2)
sns.lineplot(data=TICKER1, x="epochtime", y="open")
plt.title('Ticker1')
plt.show()

plt.subplot(1,2,1)
sns.lineplot(data=TICKER2, x="epochtime", y="open")
plt.title('Ticker2')

plt.subplot(1,2,2)
sns.lineplot(data=TICKER3, x="epochtime", y="open")
plt.title('Ticker3')
plt.show()

sns.lineplot(data=TICKER4, x="epochtime", y="open")
plt.title('Ticker4')
plt.show() 

