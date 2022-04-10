import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Read in the original dataset
failureDataset= pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/machine_failure_release.csv')
#print(failureDataset.head())

# Getting rid of redundant rows
cleanDataset = failureDataset.replace('',np.nan) # replaces '' with nan
cleanDataset = cleanDataset.replace('?',np.nan) # replaces '?' with nan
cleanDataset = cleanDataset.dropna() # drops all rows with nan

# Changing types form objects to numeric
cleanDataset['Air_temperature_K'] = cleanDataset['Air_temperature_K'].astype('float')
cleanDataset['Process_temperature_K'] = cleanDataset['Process_temperature_K'].astype('float')
cleanDataset['Rotational_speed_rpm'] = cleanDataset['Rotational_speed_rpm'].astype('int')
cleanDataset['Torque_Nm'] = cleanDataset['Torque_Nm'].astype('float')
cleanDataset['Tool_wear_min'] = cleanDataset['Tool_wear_min'].astype('int')

cleanDataset['Machine_failure'] = cleanDataset['Machine_failure'].map({'Failure': 'Yes', 'No_Failure': 'No'})

corr = cleanDataset.apply(lambda x: x.factorize()[0]).corr()

# corr.style.background_gradient(cmap='coolwarm')


f = plt.figure(figsize=(30,25))
ax = sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True,
            vmin=-1,
            vmax=1,
            linewidths=1,
            linecolor='black',
            cbar=True,
            cbar_kws={'label': 'Correlation' },
            annot_kws={"size": 70 / np.sqrt(len(corr))})


plt.xticks(range(cleanDataset.shape[1]), cleanDataset.columns, fontsize=24, rotation=30)
plt.yticks(range(cleanDataset.shape[1]), cleanDataset.columns, fontsize=24,rotation=-30)
ax.figure.axes[-1].yaxis.label.set_size(50)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=50)
plt.title('Correlation Matrix', fontsize=50)
f.savefig('/Users/seb/PycharmProjects/ACS341/courseworkSeb/graphs_outputted/correlation_figure.png')
plt.show()



#####################################################################################

cleanDataset.drop('Product_ID', inplace=True, axis=1)
cleanDataset.drop('Process_temperature_K', inplace=True, axis=1)
# One hot encoding
one_hot_type = pd.get_dummies(cleanDataset.Product_Type, prefix='Type') # add drop_first=True to save on columns
one_hot_failure = pd.get_dummies(cleanDataset.Machine_failure, prefix='Failed')

cleanDataset = pd.concat([cleanDataset, one_hot_type, one_hot_failure], axis=1)
cleanDataset.drop('Product_Type', inplace=True, axis=1)
cleanDataset.drop('Machine_failure', inplace=True, axis=1)

#print(cleanDataset.head())
# Now the product type and the machine failure are one hot encoded!!


x = cleanDataset.values
min_max_scalar = preprocessing.MinMaxScaler()
x_scaled = min_max_scalar.fit_transform(x)
scaledDataset = cleanDataset.DataFrame(x_scaled)
print(scaledDataset.head())

# Save the cleaned dataset
cleanDataset.to_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/clean_dataset.csv')


"""
print(cleanDataset.isnull().sum())#check to see if there are no nans
dfShuffled=shuffle(cleanDataset)
dfShuffled['Rotational_speed_rpm'] = dfShuffled['Rotational_speed_rpm'].astype(float)
dfShuffled['Torque [Nm]']=dfShuffled['Torque_Nm'].astype(float)

sns.barplot(x='Product Type',y='Tool wear [min]',data=cleanDataset)
sns.scatterplot(x='Torque [Nm]',y='Tool wear [min]',hue='Air temperature [K]',data=cleanDataset)
cleanDataset=cleanDataset.sort_values(by='Machine failure')#sorts the dataset over the column selected by=''
ax=sns.regplot(x='Rotational_speed_rpm',y='Torque_Nm',order=3,data=dfShuffled)
ax=sns.scatterplot(x='Rotational speed [rpm]',y='Torque [Nm]',hue='Product Type',data=dfShuffled)
ax.set(xlabel='x Label',ylabel='y Label')
plt.show()

grid = sns.FacetGrid(data=cleanDataset, col='Tool wear [min]', col_wrap=1, height=4, aspect=.75)
#
grid.map(sns.scatterplot,'Torque [Nm]')
#plt.legend(labels=['temp','priceActual'])
plt.show()
"""
