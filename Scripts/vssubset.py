import cv2
import sys
import csv
import tkinter as tk
from tkinter import filedialog

if len(sys.argv)!=2:
    print("Please specify class filter")
    sys.exit()

root=tk.Tk()
root.withdraw()

csv_ret=filedialog.askopenfilename(title="Select csv file")
if csv_ret=="":
    print("No csv file selected.")
    sys.exit()
dir_ret=filedialog.askdirectory(title="Select image location")
if dir_ret=="":
    print("No image location specified.")
    sys.exit()

with open(csv_ret,newline='\n') as csvfile:
    for row in csvfile:
        if sys.argv[1]=="NONE":
            filename=dir_ret+"/"+row.strip().split(',')[0]
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            cv2.imshow("",image)
            cv2.waitKey(0)
        else:    
            if row.strip().split(',')[1]==sys.argv[1]:
                filename=dir_ret+"/"+row.strip().split(',')[0]
                image = cv2.imread(filename, cv2.IMREAD_COLOR)
                cv2.imshow("",image)
                cv2.waitKey(0)
