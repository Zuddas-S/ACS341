import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn


fullDataFrame = pd.read_csv('/Users/seb/PycharmProjects/ACS341/LabGroupWork/DatasetDimensionReduced.csv')
fullDataFrame.head()


sns.scatterplot(x='wind_speed',y='pressure',data=fullDataFrame)
plt.show()