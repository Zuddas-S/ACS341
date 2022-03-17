from email import header
from bleach import clean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


failureDataset= pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/machine failure_release.csv')
#print(failureDataset.head())


cleanDataset=failureDataset.replace('',np.nan)
cleanDataset=cleanDataset.replace('?',np.nan)
print(cleanDataset.isnull().sum())
print(cleanDataset)


sns.scatterplot(x='Air temperature [K]',y='Rotational speed [rpm]',data=cleanDataset)
#plt.legend(labels=['temp','priceActual'])
plt.show()