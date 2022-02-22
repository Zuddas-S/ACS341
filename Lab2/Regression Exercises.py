import sys
import setup
import numpy.f2py.capi_maps
import sklearn
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv
from setup import save_fig
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


##################################################################
#Polnomial Regression



##################################################################
#Learning Curves


##################################################################
#Regularised Linear Models



##################################################################
#Logistic Regression

#Decision Boundaries
t=np.linspace(-10,10,100)
sig=1/(1+np.exp(-t))#defining our simoid function \sigma
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
plt.axis([-10, 10, -0.1, 1.1])
plt.show()
iris=datasets.load_iris()
list(iris.keys())
print(iris.DESCR) #print a description of the iris dataset.
#iris dataset contains three classes k of 50 instaces x (total of 150 instances[flowers in this case])
X=iris["data"][:,3:] #petal width
y=(iris["target"]==2).astype(np.int64)

#solver declared as "lbfgs" as this is what is used in sklearn 0.22
logReg=LogisticRegression(solver="lbfgs",random_state=42)
logReg.fit(X,y)
XNew=np.linspace(0,3,1000).reshape(-1,1)
yProbA=logReg.predict_proba(XNew)

plt.plot(XNew,yProbA[:,1],"g-",label="Iris Virginica")
plt.plot(XNew,yProbA[:,0],"b--",label="Not Iris Virginica")
plt.show()

print(logReg.predict([[1.7],[1.5]]))#outputs [1 0] stating that for an input of 1.7 we get an output
# above 0.5 hence it belongs in the class 1. The second input gives an ouput below 0.5 hence it belongs in the 0 class.
print(logReg.predict([[0.1],[1.9]]))# this gives the opposite result for the same reson.

######################################################################
#Softmax Regression
Xs=iris["data"][:,(2,3)] #petal length, petal width
ys=(iris["target"]==2).astype(np.int64)

logRegSoftmax=LogisticRegression(solver="lbfgs",C=10**10,random_state=42)
logRegSoftmax.fit(Xs,ys)

x0,x1=np.meshgrid(np.linspace(2.9,7,500).reshape(-1,1),np.linspace(0.8,2.7,200).reshape(-1,1))
XNewS=np.c_[x0.ravel(),x1.ravel()]
yProbAS=logRegSoftmax.predict_proba(XNewS)

plt.figure(figsize=(10,4))

plt.plot(Xs[ys==0,0],Xs[ys==0,1],"bs")
plt.plot(Xs[ys==1,0],Xs[ys==1,1],"g^")
zz=yProbAS[:,1].reshape(x0.shape)
contour=plt.contour(x0,x1,zz,cmap=plt.cm.brg)

leftRight=np.array([2.9,7])
boundary=-(logRegSoftmax.coef_[0][0]*leftRight+logRegSoftmax.intercept_[0]/logRegSoftmax.coef_[0][1])
plt.clabel(contour, inline=1, fontsize=12)
plt.plot(leftRight, boundary, "b--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
plt.show()


######################################################################
#
XSM=iris["data"][:,(2,3)] #getting petal length and petal width
ySM=iris["target"]

smReg=LogisticRegression(multi_class="multinomial",solver="lbfgs",C=10,random_state=42)
smReg.fit(XSM,ySM)

x2,x3=np.meshgrid(
    np.linspace(0,8,500).reshape(-1,1),
    np.linspace(0,3.5,200).reshape(-1,1),
)

XSMNew=np.c_[x2.ravel(),x3.ravel()]

yProbASM=smReg.predict_proba(XSMNew)
yPredSM=smReg.predict(XSMNew)

zz1=yProbASM[:,1].reshape(x2.shape)
zz2=yPredSM.reshape(x2.shape)

plt.figure(figsize=(10, 4))
plt.plot(XSM[ySM==2, 0], XSM[ySM==2, 1], "g^", label="Iris virginica")
plt.plot(XSM[ySM==1, 0], XSM[ySM==1, 1], "bs", label="Iris versicolor")
plt.plot(XSM[ySM==0, 0], XSM[ySM==0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
plt.contourf(x2, x3, zz2, cmap=custom_cmap)
contour = plt.contour(x2, x3, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()
############################################################################
#

def toOneHot(y):
    nClasses=y.max()+1
    m=len(y)
    yOneHot=np.zeros((m,nClasses))
    yOneHot[np.arrange(m),y]=1
    return yOneHot

def softmax(logits):
    exps=np.exp(logits)
    expSums=np.sum(exps,axis=1,keepdims=True)


#where does yTrainOneHot come in??
"""
def train(xTrain,yTrain):
    nInputs=xTrain.shape[1]
    nOutputs=len(np.unique(yTrain))

    eta=0.01
    nIteration=5001
    m=len(xTrain)
    epsilon=1e-7

    Theta=np.random.randn(nInputs,nOutputs)
    for iteration in range(nIteration):
        logits=xTrain.dot(Theta)
        yProbA=softmax(logits)
        loss=-np.mean(np.sum(yTrainOneHot*np.log(yProbA+epsilon),axis=1))
        #loss value
        error=yProbA-yTrainOneHot
        if iteration%500==0:
            print(iteration,loss)
        gradients=1/m*xTrain.T.dot(error)
        Theta=Theta-eta*gradients
    return Theta
"""