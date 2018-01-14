import tensorflow as tf
import numpy as np

def initialize_variables():
    X=tf.placeholder(tf.float32,shape=[32*32*3,None])
    Y=tf.placeholder(tf.float32,shape=[1,None])
    W1=tf.get_variable("W1",[128,32*32*3],initializer=tf.contrib.layers.xavier_initializer())
    b1=tf.get_variable("b1",[128,1],initializer=tf.zeros_initializer())
    W2=tf.get_variable("W2",[64,128],initializer=tf.contrib.layers.xavier_initializer())
    b2=tf.get_variable("b2",[64,1],initializer=tf.zeros_initializer())
    W3=tf.get_variable("W3",[16,64],initializer=tf.contrib.layers.xavier_initializer())
    b3=tf.get_variable("b3",[16,1],initializer=tf.zeros_initializer())
    W4=tf.get_variable("W4",[1,16],initializer=tf.contrib.layers.xavier_initializer())
    b4=tf.get_variable("b4",[1,1],initializer=tf.zeros_initializer())
    params={}
    params['X']=X
    params['Y']=Y
    params['W1']=W1
    params['W2']=W2
    params['W3']=W3
    params['W4']=W4
    params['b1']=b1
    params['b2']=b2
    params['b3']=b3
    params['b4']=b4
    return params

def forward_prop(X,params):
    W1=params['W1']
    W2=params['W2']
    W3=params['W3']
    W4=params['W4']
    b1=params['b1']
    b2=params['b2']
    b3=params['b3']
    b4=params['b4']
    Z1=tf.matmul(W1,X)+b1
    A1=tf.nn.relu(Z1)
    Z2=tf.matmul(W2,A1)+b2
    A2=tf.nn.relu(Z2)
    Z3=tf.matmul(W3,A2)+b3
    A3=tf.nn.relu(Z3)
    Z4=tf.matmul(W4,A3)+b4
    A4=tf.sigmoid(Z4)
    return A4

def cost_func(Y,Yhat):
    cost=-tf.reduce_mean(tf.multiply(Y,tf.log(Yhat))+tf.multiply(1-Y,tf.log(1-Yhat)))
    return cost

def get_accuracy(Y,Yhat):
    m=Y.shape[1]
    wrongind=[]
    wrongcnt=0
    for i in range(m):
        if np.round(Yhat[0,i])!=Y[0,i]:
            wrongcnt=wrongcnt+1
            wrongind.append(i)
    return 1-wrongcnt/m,wrongind
