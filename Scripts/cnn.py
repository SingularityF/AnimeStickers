import tensorflow as tf
import numpy as np

def initialize_variables():
    X=tf.placeholder(tf.float32,shape=[None,32,32,3])
    Y=tf.placeholder(tf.float32,shape=[1,None])
    W1=tf.get_variable("W1",[3,3,3,16],initializer=tf.contrib.layers.xavier_initializer())
    b1=tf.get_variable("b1",[16],initializer=tf.zeros_initializer())
    W2=tf.get_variable("W2",[3,3,16,32],initializer=tf.contrib.layers.xavier_initializer())
    b2=tf.get_variable("b2",[32],initializer=tf.zeros_initializer())
    W3=tf.get_variable("W3",[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer())
    b3=tf.get_variable("b3",[64],initializer=tf.zeros_initializer())
    W4=tf.get_variable("W4",[128,1024],initializer=tf.contrib.layers.xavier_initializer())
    b4=tf.get_variable("b4",[128,1],initializer=tf.zeros_initializer())
    W5=tf.get_variable("W5",[16,128],initializer=tf.contrib.layers.xavier_initializer())
    b5=tf.get_variable("b5",[16,1],initializer=tf.zeros_initializer())
    W6=tf.get_variable("W6",[1,16],initializer=tf.contrib.layers.xavier_initializer())
    b6=tf.get_variable("b6",[1,1],initializer=tf.zeros_initializer())
    params={}
    params["X"]=X
    params["Y"]=Y
    params["W1"]=W1
    params["W2"]=W2
    params["W3"]=W3
    params["W4"]=W4
    params["W5"]=W5
    params["W6"]=W6
    params["b1"]=b1
    params["b2"]=b2
    params["b3"]=b3
    params["b4"]=b4
    params["b5"]=b5
    params["b6"]=b6
    return params

def forward_prop(X,params):
    W1=params["W1"]
    W2=params["W2"]
    W3=params["W3"]
    W4=params["W4"]
    W5=params["W5"]
    W6=params["W6"]
    b1=params["b1"]
    b2=params["b2"]
    b3=params["b3"]
    b4=params["b4"]
    b5=params["b5"]
    b6=params["b6"]
    Z1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
    Z1=tf.nn.bias_add(Z1,b1)
    A1=tf.nn.relu(Z1)
    P1=tf.nn.max_pool(A1,[1,2,2,1],[1,2,2,1],padding="VALID")
    Z2=tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")
    Z2=tf.nn.bias_add(Z2,b2)
    A2=tf.nn.relu(Z2)
    P2=tf.nn.max_pool(A2,[1,2,2,1],[1,2,2,1],padding="VALID")
    Z3=tf.nn.conv2d(P2,W3,strides=[1,1,1,1],padding="SAME")
    Z3=tf.nn.bias_add(Z3,b3)
    A3=tf.nn.relu(Z3)
    P3=tf.nn.max_pool(A3,[1,2,2,1],[1,2,2,1],padding="VALID")
    F1=tf.layers.flatten(P3)
    F1=tf.transpose(F1)
    Z4=tf.matmul(W4,F1)+b4
    A4=tf.nn.relu(Z4)
    Z5=tf.matmul(W5,A4)+b5
    A5=tf.nn.relu(Z5)
    Z6=tf.matmul(W6,A5)+b6
    A6=tf.sigmoid(Z6)
    return A6

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
