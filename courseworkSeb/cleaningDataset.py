import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

failureDataset= pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/machine failure_release.csv')
#print(failureDataset.head())

cleanDataset=failureDataset.replace('',np.nan)#replaces '' with nan
cleanDataset=cleanDataset.replace('?',np.nan)#replaces '?' with nan
cleanDataset=cleanDataset.dropna()#drops all rows with nan

print(cleanDataset.isnull().sum())#check to see if there are no nans
dfShuffled=shuffle(cleanDataset)
dfShuffled['Rotational speed [rpm]'] = dfShuffled['Rotational speed [rpm]'].astype(float)
dfShuffled['Torque [Nm]']=dfShuffled['Torque [Nm]'].astype(float)
print(dfShuffled)


#sns.barplot(x='Product Type',y='Tool wear [min]',data=cleanDataset)
#sns.scatterplot(x='Torque [Nm]',y='Tool wear [min]',hue='Air temperature [K]',data=cleanDataset)


#cleanDataset=cleanDataset.sort_values(by='Machine failure')#sorts the dataset over the column selected by=''
ax=sns.regplot(x='Rotational speed [rpm]',y='Torque [Nm]',order=3,data=dfShuffled)
#ax=sns.scatterplot(x='Rotational speed [rpm]',y='Torque [Nm]',hue='Product Type',data=dfShuffled)
ax.set(xlabel='x Label',ylabel='y Label')
plt.show()


"""
grid = sns.FacetGrid(data=cleanDataset, col='Tool wear [min]', col_wrap=1, height=4, aspect=.75)
#
grid.map(sns.scatterplot,'Torque [Nm]')
#plt.legend(labels=['temp','priceActual'])
plt.show()
"""
