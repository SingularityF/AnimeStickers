import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import os
import sys
from regularnn import *
from readdata import *

root=tk.Tk()
root.withdraw()

#Get training set
csv_ret,dir_ret=open_selection()
X_train,Y_train,_=read_data_flat(csv_ret,dir_ret)
#End get training set

#Train neural network
params=initialize_variables()
Y=params['Y']
X=params['X']
Yhat=forward_prop(X,params)
cost=cost_func(Y,Yhat)
optimizer=tf.train.AdamOptimizer(learning_rate=.002).minimize(cost)
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for i in range(300):
    _,num_cost=sess.run([optimizer,cost],feed_dict={X:X_train,Y:Y_train})
    print(num_cost)
Yhat_num=sess.run(Yhat,feed_dict={X:X_train,Y:Y_train})
#Neural network trained

#Calculate training set accuracy
accu,_=get_accuracy(Y_train,Yhat_num)
print("You got "+str(100*accu)+"% right!",flush=True)
#END training set accuracy

#Get test set
csv_ret,dir_ret=open_selection()
X_test,Y_test,filenames=read_data_flat(csv_ret,dir_ret)
#END Get test set

#Test on test set
Yhat_num=sess.run(Yhat,feed_dict={X:X_test,Y:Y_test})
accu,wrongind=get_accuracy(Y_test,Yhat_num)
print("You got "+str(100*accu)+"% right!")
#END test on test set

wronglist=[filenames[i] for i in wrongind]
csv_ret=filedialog.asksaveasfilename(title="Select csv output location",defaultextension=".csv")
if csv_ret=="":
    print("No csv file selected.")
    sys.exit()

with open(csv_ret,"w") as fp:
    writer=csv.writer(fp,lineterminator='\n')
    for i in range(len(wronglist)):
        writer.writerow([os.path.basename(wronglist[i])])

sess.close()
