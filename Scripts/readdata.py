import cv2
import tkinter as tk
from tkinter import filedialog
import csv
import numpy as np
import sys


def open_selection():
    csv_ret=filedialog.askopenfilename(title="Select csv file")
    if csv_ret=="":
        print("No csv file selected.")
        sys.exit()
    dir_ret=filedialog.askdirectory(title="Select image location")
    if dir_ret=="":
        print("No image location specified.")
        sys.exit()
    return csv_ret,dir_ret

def read_data_flat(csv_ret,dir_ret):
    filenames=[]
    Ydat=np.empty((1,0),dtype=float)
    Xdat=np.empty((32*32*3,0),dtype=float)
    with open(csv_ret,newline='\n') as csvfile:
        for row in csvfile:
            classno=row.strip().split(',')[1]
            classno=float(classno)
            classno=np.array([[classno]])
            filename=dir_ret+"/"+row.strip().split(',')[0]
            filenames.append(filename)
            Ydat=np.hstack((Ydat,classno))
    for file in filenames:
        imorig=cv2.imread(file, cv2.IMREAD_COLOR)
        im=cv2.resize(imorig,(32,32),interpolation=cv2.INTER_CUBIC)
        data=np.array(im).astype(float)
        data=data/255
        flatdata=data.flatten()
        flatdata=flatdata.reshape(32*32*3,1)
        Xdat=np.hstack((Xdat,flatdata))
    return Xdat,Ydat,filenames

def read_data(csv_ret,dir_ret):
    filenames=[]
    Ydat=np.empty((1,0),dtype=float)
    Xdat=np.empty((0,32,32,3),dtype=float)
    with open(csv_ret,newline='\n') as csvfile:
        for row in csvfile:
            classno=row.strip().split(',')[1]
            classno=float(classno)
            classno=np.array([[classno]])
            filename=dir_ret+"/"+row.strip().split(',')[0]
            filenames.append(filename)
            Ydat=np.hstack((Ydat,classno))
    for file in filenames:
        imorig=cv2.imread(file, cv2.IMREAD_COLOR)
        im=cv2.resize(imorig,(32,32),interpolation=cv2.INTER_CUBIC)
        data=np.array(im).astype(float)
        data=data/255
        data=data.reshape(1,32,32,3)
        Xdat=np.append(Xdat,data,axis=0)
    return Xdat,Ydat,filenames
