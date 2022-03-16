import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

fullDataFrame = pd.read_csv('/Users/seb/PycharmProjects/ACS341/LabGroupWork/Dataset_cleaned_FINAL.csv',low_memory=False)
fullDataFrame.head()

#print(fullDataFrame.isnull().sum()==0) test to see whether the data is clean


#plotting will not work without clean data. 
#sns.scatterplot(x='rain_1h',y='pressure',data=fullDataFrame)
#plt.show()


sns.scatterplot(x='time_of_day',y='temp',hue='temp',data=fullDataFrame)
#plt.legend(labels=['temp','priceActual'])
plt.show()