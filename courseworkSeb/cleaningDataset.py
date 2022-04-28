"""

Sebastiano Zuddas 2022
Program to clean the dataset provided using pandas.

"""

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

# Getting rid of redundant rows
cleanDataset = failureDataset.replace('', np.nan)  # replaces '' with nan
cleanDataset = cleanDataset.replace('?', np.nan)  # replaces '?' with nan
cleanDataset = cleanDataset.dropna()  # drops all rows with nan

# Changing types form objects to numeric
cleanDataset['Air_temperature_K'] = cleanDataset['Air_temperature_K'].astype('float')
cleanDataset['Process_temperature_K'] = cleanDataset['Process_temperature_K'].astype('float')
cleanDataset['Rotational_speed_rpm'] = cleanDataset['Rotational_speed_rpm'].astype('int')
cleanDataset['Torque_Nm'] = cleanDataset['Torque_Nm'].astype('float')
cleanDataset['Tool_wear_min'] = cleanDataset['Tool_wear_min'].astype('int')
cleanDataset['Machine_failure'] = cleanDataset['Machine_failure'].map({'Failure': 'Yes', 'No_Failure': 'No'})

#####################################################################################
# Plotting correlation matrix
corr = cleanDataset.apply(lambda x: x.factorize()[0]).corr()
# lambda fcn to see correlation between non numeric features

f = plt.figure(figsize=(30, 25))
ax = sns.heatmap(corr,
                 xticklabels=corr.columns.values,
                 yticklabels=corr.columns.values,
                 annot=True,
                 vmin=-1,
                 vmax=1,
                 linewidths=1,
                 linecolor='black',
                 cbar=True,
                 cbar_kws={'label': 'Correlation'},
                 annot_kws={"size": 70 / np.sqrt(len(corr))})

plt.xticks(range(cleanDataset.shape[1]), cleanDataset.columns, fontsize=24, rotation=30)
plt.yticks(range(cleanDataset.shape[1]), cleanDataset.columns, fontsize=24, rotation=-30)
ax.figure.axes[-1].yaxis.label.set_size(50)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=50)
plt.title('Correlation Matrix', fontsize=50)
f.savefig('/Users/seb/PycharmProjects/ACS341/courseworkSeb/graphs_outputted/correlation_figure.png')

#####################################################################################
# Processing

cleanDataset.drop('Product_ID', inplace=True, axis=1)
cleanDataset.drop('Process_temperature_K', inplace=True, axis=1)
# One hot encoding
one_hot_type = pd.get_dummies(cleanDataset.Product_Type, prefix='Type') # add drop_first=True to save on columns
one_hot_failure = pd.get_dummies(cleanDataset.Machine_failure, prefix='Failed')

cleanDataset = pd.concat([cleanDataset, one_hot_type, one_hot_failure], axis=1)
cleanDataset.drop('Product_Type', inplace=True, axis=1)
cleanDataset.drop('Machine_failure', inplace=True, axis=1)


x = cleanDataset
min_max_scalar = preprocessing.MinMaxScaler()
x_scaled = min_max_scalar.fit_transform(x)
scaledDataset = pd.DataFrame(x_scaled)
scaledDataset.columns = ['Air_temperature_K',  'Rotational_speed_rpm',  'Torque_Nm',  'Tool_wear_min', 'Type_H',  'Type_L',  'Type_M',  'Failed_Yes']

# Save the cleaned dataset
# cleanDataset.to_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/clean_dataset.csv')
# scaledDataset.to_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/scaled_dataset.csv')
# plt.show()

