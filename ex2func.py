#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib.pyplot as plt

def plotData(X,y):
    plt.figure(figsize=(10,6))
    pos=np.where(y==1)
    neg=np.where(y==0)
    plt.plot(X[pos,0],X[pos,1],'k+')
    plt.plot(X[neg,0],X[neg,1],'yo')
    plt.grid(True)


# In[2]:


def costFunction(theta,X,y):
    m,n = X.shape
    J=0
    grad=np.zeros(np.size(theta))
    prediction=sigmoid(X.dot(theta))
    J=1/m*sum(-y*np.log(prediction)-(1-y)*np.log(1-prediction))
    error=prediction-y
    delta=1/m*error.T.dot(X)
    grad=delta
    return J, grad


# In[3]:


def myCost(theta,X,y):
    m,n=X.shape
    predictions=X.dot(theta)
    sig_term=sigmoid(predictions)
    J=1/m*np.sum(-y*np.log(sig_term)-(1-y)*np.log(1-sig_term))
    return J


# In[4]:


def myGradient(theta,X,y):
    m,n=np.shape(X)
    predictions=sigmoid(X.dot(theta))
    errors=predictions-y
    delta=1/m*errors.T.dot(X)
    grad=delta
    return grad


# In[5]:


def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g


# In[6]:


def plotDecisionBoundary(theta,X,y):
    plotData(X[:,1:3],y)
    
    if np.size(X,1)<=3:
        plot_x=np.array([np.min(X[:,1])-2,np.max(X[:,1]+2)])
        plot_y=(-1/theta[2])*(theta[1]*plot_x+theta[0])
        plt.plot(plot_x,plot_y)
        plt.legend(('Admitted','Not admitted'), loc='upper right')
    else:
        u=np.linspace(-1,1.5,50)
        v=np.linspace(-1,1.5,50)
        z=np.zeros((np.size(u),np.size(v)))
        
        for i in np.arange(np.size(u)):
            for j in np.arange(np.size(v)):
                z[i,j]=np.sum(mapFeature(u[i],v[j])*theta)
        
        z=z.T
        plt.contour(u,v,z)


# In[7]:


def predict01(theta,X):
    m=X.shape[0]
    p=np.zeros(m)
    p=sigmoid(X.dot(theta))>=0.5;
    return p


# In[9]:


def probofobs(theta, X):
    return sigmoid((X.dot(theta)))



def mapFeature(x1,x2):
    degree=6
    m=np.size(x1)
    out=np.ones(m)
    res=[]
    res.append(out)
    for i in np.arange(1, degree+1):
        for j in np.arange(0,i+1):
            out1=(x1**(i-j))*(x2**j)
            res.append(out1)
    return np.array(res).T


def costFunctionReg(theta,X,y,s_lambda):
    m,n=X.shape
    J=0
    grad=np.zeros(np.size(theta))
    sum_theta_square=np.sum(theta[1:]**2)
    
    X_theta = X.dot(theta)
    prediction = sigmoid(X_theta)
    
    J=1/m*np.sum(-y*np.log(prediction)-(1-y)*np.log(1-prediction))+s_lambda/(2*m)*sum_theta_square
    error=prediction - y
    delta=1/m*error.T.dot(X)
    
    temp=theta
    temp[0]=0
    regular=(s_lambda/m)*temp
    grad = delta + regular
    return J, grad