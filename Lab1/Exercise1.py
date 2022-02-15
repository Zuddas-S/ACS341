import sys

import numpy.f2py.capi_maps
import sklearn
import numpy as np
import os
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "/Users/seb/PycharmProjects/ACS341/Lab1"
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "../images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


x=2*np.random.rand(100,1)
y=4+3*x+np.random.rand(100,1)
plt.plot(x,y,"b.")
plt.xlabel("$x_1$",fontsize=17)
plt.ylabel("$y$",rotation=0,fontsize=18)
plt.show()
save_fig("generatedDataPlot")

xb=np.c_[np.ones((100,1)),x] #adds x0=1 to each instance
theta_best=np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
theta_best

xnew=np.array([[0],[2]])
xnewb=np.c_[np.ones((2,1)),xnew]
ypredict=xnewb.dot(theta_best)
ypredict

plt.plot(xnew, ypredict, "r-")
plt.plot(x, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

linReg=LinearRegression()
linReg.fit(x,y)
linReg.intercept_,linReg.coef_
linReg.predict(xnew)
#Because the linear regression class is based off scipy.linalg.lstsq() you can directly call:


thetaBestSvd,residuals,rank,s=np.linalg.lstsq(xb,y,rcond=1e-6)# note the 1e-6 is 1x10^-6
#computes ls through the pinv directly
np.linalg.pinv(xb).dot(y)

#############################################################

with open('/Users/seb/PycharmProjects/ACS341/Lab1/data_quiz_1.csv') as quizOneData:
    reader=csv.reader(quizOneData)
    next(reader)
    quizOneList=list(csv.reader(quizOneData))

print(quizOneList)
quizOneListT=np.array(quizOneList).T.tolist()
print(quizOneListT)


xArray,yArray=[],[]

xlist=[]

for i in quizOneList:
    x[i]=quizOneListT[i]
print(x)

