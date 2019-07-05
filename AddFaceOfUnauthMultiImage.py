from urllib.request import urlopen
import cv2
import os
import pickle
import numpy as np
import sqlite3
from PIL import Image
import datetime

path='RecordPicture'
num = 0

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_default.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('Recognizer/trainningData.yml')
id=0
font=cv2.FONT_HERSHEY_SIMPLEX

def getProfile(id):
    conn=sqlite3.connect("facebase.db")
    cmd="SELECT * FROM People WHERE Id="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
for imagePath in imagePaths:
    image=str(os.path.split(imagePath)[1])
    imag=cv2.imread('RecordPicture/'+image,1)
    gray1=cv2.imread('RecordPicture/'+image,0)
    r = 720.0 / imag.shape[1]
    dim = (640, int(imag.shape[0] * r))
    img = cv2.resize(imag, dim, interpolation = cv2.INTER_AREA)
    r1 = 700.0 / gray1.shape[1]
    dim1 = (640, int(gray1.shape[0] * r))
    gray = cv2.resize(gray1, dim, interpolation = cv2.INTER_AREA)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        #cv2.rectangle(img, (x,y), (end_cord_x,end_cord_y), color, stroke)
        #cv2.rectangle(img, (x,y), (x+w,y+h),(112, 255, 1), 2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])   # get ID and Confidence, Less is good
        profile=getProfile(id)
        if conf < 70:
            cv2.rectangle(img, (x,y), (x+w,y+h),(170,170,170),1)
            cv2.putText(img,str(profile[1]),(x+10,y+h+20),font, .65, (64,204,46),1)
        else:
            num = num+1
            cv2.rectangle(img, (x,y), (x+w,y+h),(27,133,255),1)
            cv2.putText(img,'Unauthorised'+'_'+str(num), (x,y+h+15),font, .55,(54,65,255),1)
            CT = datetime.datetime.now()
            cv2.imwrite("Unauthorised/"+str(num)+str(CT.strftime("%Y-%m-%d-%H-%M-%S"))+".jpg",gray[y:y+h,x:x+w]) #add picture of faces in Folder Unauthorised
    a=cv2.imshow('View',img)
    cv2.waitKey(0)
    cv2.destroyWindow(a)

cv2.destroyAllWindows()
