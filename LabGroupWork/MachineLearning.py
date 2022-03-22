from email import header
from http.cookiejar import FileCookieJar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

fullDataFrame = pd.read_csv('/Users/seb/PycharmProjects/ACS341/LabGroupWork/Dataset_cleaned_FINAL.csv',low_memory=False)
fullDataFrame.head()

print(fullDataFrame.isnull().sum())# test to see whether the data is clean

energy_data=fullDataFrame[['generationFossilBrownCoal_lignite','generationFossilGas','generationBiomass','generationFossilOil','time_of_day','Epoch_time','humidity','generationSolar','generationWaste','forecastSolarDayAhead','forecastWindOnshoreDayAhead','totalLoadActual','generationWindOnshore','generationOther','generationNuclear','generationHydroRun_of_riverAndPoundage','generationHydroWaterReservoir','totalLoadForecast','generationOtherRenewable','generationHydroPumpedStorageConsumption','generationFossilHardCoal']]
price_data=fullDataFrame[['priceDayAhead','priceActual']]

print((energy_data))


shuffled_data=shuffle(fullDataFrame)


#plotting will not work without clean data. 
#sns.scatterplot(x='rain_1h',y='pressure',data=shuffled_data)
#plt.show()
#


"""



shuffled_data['temp'] = shuffled_data['temp'].astype(float)
shuffled_data['generationFossilBrownCoal_lignite'] = shuffled_data['generationFossilBrownCoal_lignite'].astype(float)

columns=2

fig,axs=plt.subplots(ncols=columns)#make a figure with 2 columns for two plots
sns.regplot(x='time_of_day',y='temp',order=3,data=shuffled_data,ax=axs[0])
sns.scatterplot(x='time_of_day',y='temp',hue='generationFossilBrownCoal_lignite',data=fullDataFrame,ax=axs[1])
#plt.legend(labels=['temp','priceActual'])
plt.show()"""