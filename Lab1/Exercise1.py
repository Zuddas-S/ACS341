#Setup

import setup
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


x=2*np.random.rand(100,1)
y=4+3*x+np.random.rand(100,1)
plt.plot(x,y,"b.")
plt.xlabel("$x_1$",fontsize=17)
plt.ylabel("$y$",rotation=0,fontsize=18)
plt.show()
setup.save_fig("generatedDataPlot")

xb=np.c_[np.ones((100,1)),x] #adds x0=1 to each instance
theta_best=np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y) #this is the pinvPhiY

xnew=np.array([[0],[2]])
xnewb=np.c_[np.ones((2,1)),xnew]
ypredict=xnewb.dot(theta_best)

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

##########################################################################################################################
#Exercise 1
xList=[]
yList=[]

with open('/Users/seb/PycharmProjects/ACS341/Lab1/data_quiz_1.csv') as quizOneData:
    reader=csv.reader(quizOneData,delimiter=',')
    readerT = np.array(reader).T.tolist()#transpose of Reader
    for rows in reader:
        #note these are given as lists
        xList.append(float(rows[1])) #x column is the second one (1 from 0)
        yList.append(float(rows[0])) #y column is the first one

plt.plot(xList,yList,'b.')
plt.show()

xList=xList
yList=yList

xListB=np.c_[np.ones((len(xList),1)),xList]
betaHat=np.linalg.inv(xListB.T.dot(xListB)).dot(xListB.T).dot(yList) #generate the coeffs by minimising loss function

xListNew=np.array([[np.min(xList)],[np.max(xList)]])#We give the new array the boundaries from min of xList to max of xList
xListNewB=np.c_[np.ones((2,1)),xListNew] #Create some ones
yListPredict=xListNewB.dot(betaHat) #element wise multiplication using the values from betahat to get the predicted outputs


plt.plot(xList,yList,'b.')
plt.plot(xListNew,yListPredict,'-r')
#plot the two on the same graph.
plt.show()

#Now we want to actually perform an error minimisation using linearregression function
from numpy import reshape

twoDimensionalData=xList+yList
xArray=np.array(xList)
yArray=np.array(yList)
xArray=xArray.reshape(-1,1)
yArray=yArray.reshape(-1,1)
linReg=LinearRegression()
#fit and predict methods need 2D arrays!!
#linReg.fit(xArray,yArray)
#linReg.intercept_,linReg.coef_
#yExPred=linReg.predict(xListNew)

#plt.scatter(xList,yList,'.b')
#plt.plot(np.array([[0],[len(yExPred)]]),yExPred,'-r')
#plt.show()
