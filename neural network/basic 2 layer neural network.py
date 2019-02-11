import numpy as np
import matplotlib.pyplot as plt
import random
X=np.array([[4,1],[5,2],[6,3]])
Y=np.array([[1,1]])
n_x=X.shape[0]
n_h=4
np.random.seed(2)
w1=np.random.randn(n_h,n_x)
b1=np.zeros(shape=(n_h,1))
w2=np.random.randn(1,n_h)
b2=np.zeros(shape=(1,1))
parameters={'w1':w1,'w2':w2,'b1':b1,'b2':b2}
m=X.shape[1]
def compute1(parameters,X,Y):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    z1 = np.dot(w1,X)+b1
    a1 = (np.exp(z1)-np.exp(-z1))/(np.exp(z1)+np.exp(-z1))
    z2 = np.dot(w2,a1)+b2
    a2 = (1/(1+np.exp(-z2)))
    cache = {'z1':z1,'z2':z2,'a1':a1,'a2':a2}
    loss = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = -np.sum(loss)/m
    return cache,cost
def backpropogation(parameters,cache,X,Y):
    w1 = parameters['w1']
    w2 = parameters['w2']
    a1 = cache['a1']
    a2 = cache['a2']
    dz2 = a2-Y
    dw2 = (1/m)*np.dot(dz2,a1.T)
    db2 = (1/m)*np.sum(dz2,axis=1,keepdims=True)
    dz1 = np.multiply(np.dot(w2.T,dz2),1-np.power(a1,2))
    dw1 = (1/m)*np.dot(dz1,X.T)
    db1 = (1/m)*np.sum(dz1,axis=1,keepdims=True)
    grades={'dw1':dw1,'dw2':dw2,'db1':db1,'db2':db2}
    return grades;
def update(grades,parameters,learning_rate=1.2):
    dw1=grades['dw1']
    dw2=grades['dw2']
    db1=grades['db1']
    db2=grades['db2']
    w1=parameters['w1']
    w2=parameters['w2']
    b1=parameters['b1']
    b2=parameters['b2']
    w1=w1-learning_rate*dw1
    w2=w2-learning_rate*dw2
    b1=b1-learning_rate*db1
    b2=b2-learning_rate*db2
    parameters={'w1':w1,'w2':w2,'b1':b1,'b2':b2}
    return parameters
def NeuralNetwork(X,Y,no_of_iteration,count=False):
    np.random.seed(3)
    n_x=X.shape[0]
    n_h=4
    w1=np.random.randn(n_h,n_x)
    b1=np.zeros(shape=(n_h,1))
    w2=np.random.randn(1,n_h)
    b2=np.zeros(shape=(1,1))
    parameters={'w1':w1,'w2':w2,'b1':b1,'b2':b2}
    m=X.shape[1]
    for i in range(0,no_of_iteration):
        cache,cost=compute1(parameters,X,Y)
        grades=backpropogation(parameters,cache,X,Y)
        parameters=update(grades,parameters,learning_rate=1.2)
        print("cost after iteration %i: %f" %(i,cost))
    return parameters

NeuralNetwork(X,Y,20,count=True)
print(parameters)
print("gangotrinvents")
