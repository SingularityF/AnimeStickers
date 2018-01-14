import cv2
import sys
import os.path
import tkinter as tk
from tkinter import filedialog

num=1
cascade_file = "lbpcascade_animeface.xml"
if not os.path.isfile(cascade_file):
    raise RuntimeError("%s: not found" % cascade_file)

cascade = cv2.CascadeClassifier(cascade_file)
root=tk.Tk()
root.withdraw()
file_ret=filedialog.askopenfilenames(title="Select images to detect")
if len(file_ret)==0:
    print("No files selected.")
    sys.exit()
save_ret=filedialog.askdirectory(title="Select save location")
if save_ret=="":
    print("No save location specified.")
    sys.exit()
files=root.tk.splitlist(file_ret)
print(save_ret)

def detect(filename):
    global num
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (128, 128)
                                     )
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #cv2.imshow("AnimeFaceDetect", image)
    #cv2.waitKey(0)
    for (x, y, w, h) in faces:
        subimg=image[y:y+h,x:x+w]
        #resizeimg=cv2.resize(subimg,(256,256),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(save_ret+"/"+"face"+str(num)+".jpg",subimg)
        num=num+1

for file in files:
    detect(file)
