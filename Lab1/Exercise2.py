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

thetaPathSGD=[]
m=len(xb)
np.random.seed(42)

nEpochs=50
t0,t1=5,50 # Learning Schedule Hyperparameters
def learningSchedule(t):
    return t0/(t+t1)
theta=np.random.randn(2,1) # random initialisation

for epoch in range(nEpochs):
    for i in range(m):
        if epoch==0 and i<20:
            yPredict=xnewb.dot(theta)
            style='b-'if i>0 else 'r--' # first iteration make a red dashed line
            plt.plot(xnew,yPredict,style)
        randomIndex = np.random.randint(m)
        xi = xb[randomIndex:randomIndex + 1]
        yi = y[randomIndex:randomIndex + 1]
        gradients=2*xi.T.dot(xi.dot(theta)-yi)
        eta=learningSchedule(epoch*m+i)
        theta=theta-eta*gradients
        thetaPathSGD.append(theta)

plt.plot(x,y,'b.')
plt.xlabel('$x_1$',fontsize=18)
plt.ylabel('$y$',rotation=0,fontsize=18)
plt.axis([0,2,0,15])
setup.save_fig("Stochastic Gradient Descent")
plt.show()
##############################################################################################
#

thetaPathMBGD=[]
nIterations=50
miniBatchSize=20
np.random.seed(42)
theta=np.random.randn(2,1)
t,t0,t1=0,200,1000

## We use the learning scedule function already defined above

for epoch in range(nIterations):
    shuffledIndices=np.random.permutation(m)
    xbShuffled=xb[shuffledIndices]
    yShuffled=y[shuffledIndices]
    for i in range(0,m,miniBatchSize):
        t+=1
        xi=xbShuffled[i:i+miniBatchSize]
        yi=yShuffled[i:i+miniBatchSize]
        gradients=2/miniBatchSize * xi.T.dot(xi.dot(theta)-yi)
        eta=learningSchedule(t)
        theta=theta-eta*gradients
        thetaPathMBGD.append(theta)

    ### putting all grad desc. algos together.

thetaPathBGD=np.array(thetaPathBGD)
thetaPathSGD=np.array(thetaPathSGD)
thetaPathMBGD=np.array(thetaPathMBGD)
plt.figure(figsize=(7,4))
plt.plot(thetaPathSGD[:,0],thetaPathSGD[:,1],'r-s',linewidth=1,label="Stochastic")
plt.plot(thetaPathMBGD[:,0],thetaPathMBGD[:,1],'b-o',linewidth=1,label="Mini-Batch")
plt.plot(thetaPathBGD[:,0],thetaPathBGD[:,1],'g-+',linewidth=1,label="Batch")
plt.legend(loc='upper left',fontsize=12)
plt.xlabel(r'$\theta_0$',fontsize=10)
plt.ylabel(r'$\theta_1$',fontsize=10,rotation=0)
plt.axis([2.5,5,2.3,4.5])
setup.save_fig('Comparison of Gradient Descent Methods')
plt.show()
################################################################################################################
#Polynomial Regression


