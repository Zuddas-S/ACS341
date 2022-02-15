#Setup
import sys
import setup
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
# Code BELOW

########################################################################################################
eta=0.1 #defining our learning rate
nIterations=1000
m=100

x=2*np.random.rand(100,1)
y=4+3*x+np.random.rand(100,1)

xb=np.c_[np.ones((100,1)),x] #adds x0=1 to each instance
theta=np.random.randn(2,1) #random initialisation
thetaPathBGD=[]
for iteration in range(nIterations):
    gradients =2/m *xb.T.dot(xb.dot(theta)-y)#this is the actual gradient descent
    theta = theta-eta*gradients #this gets the iterative values of theta, this goes around 1000 times as defined
#theta

xnew=np.array([[0],[2]])#we take the same as in exercise 1
xnewb=np.c_[np.ones((2,1)),xnew]
xnewb.dot(theta)

#this is a function that gives the
def plotGradientDescent(theta,eta,thetaPath=None,nIerations=1000):
    m=len(xb)
    plt.plot(x,y,'b.')#plot the original dataset to which we're looking to make a model for
    for iteration in range(nIterations):
        if iteration<10:
            yPred=xnewb.dot(theta)
            style="b-" if iteration>0 else "r--"
            plt.plot(xnew,yPred,style)
        gradients=2/m*xb.T.dot(xb.dot(theta)-y)
        theta=theta-eta*gradients
        if thetaPath is not None:
            thetaPath.append(theta)
    plt.xlabel("$x_1$",fontsize=18)
    plt.axis([0,2,0,15])
    plt.title(r"$\eta={}$".format(eta),fontsize=16)

np.random.seed(42)
theta=np.random.randn(2,1)#for a random initialisation

plt.figure(figsize=(10,4))#making a figure
plt.subplot(131);plotGradientDescent(theta,eta=0.02) #we see that learning rate too small takes many iterations - slow
plt.ylabel("$y$",rotation=0,fontsize=18)#we only need one y on the LHS to represent all of our figs
plt.subplot(132);plotGradientDescent(theta,eta=0.1,thetaPath=thetaPathBGD)#just right learning rate, fast but accurate and finds correct \theta
plt.subplot(133);plotGradientDescent(theta,eta=0.5)#too high learning rate, line jumps about too much.
setup.save_fig("Gradient Descent Plot")#save the plot using the funciton made in setup
plt.show()

####################################################################################
#Stochastic gradient descent





