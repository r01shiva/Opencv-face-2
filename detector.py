import numpy as np   #to convert images into arrays
import cv2           #to use cv2 library
import sqlite3       #add database
import datetime
import time

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
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


while(True):
    ret, img= cap.read() #read camera colored video for user only
    ret, img2 = cap.read() #read camera for UnPicture
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert image for cv2 to Gray because it is working on Gray image only
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        #cv2.rectangle(img, (x,y), (end_cord_x,end_cord_y), color, stroke)
        #cv2.rectangle(img, (x,y), (x+w,y+h),(112, 255, 1), 2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])   # get ID and Confidence, Less is good
        profile=getProfile(id) #call for id to look into database
        if conf < 70:
            cv2.rectangle(img, (x,y), (x+w,y+h),(170,170,170),1) # create a rectangle around faces color gray
            cv2.putText(img,str(profile[1]),(x+10,y+h+20),font, .65, (64,204,46),1) #put text of color Lime
        else:
            cv2.rectangle(img, (x,y), (x+w,y+h),(27,133,255),1)
            cv2.putText(img, 'Unauthorised', (x,y+h+15),font, .55,(54,65,255),1)
            CT = datetime.datetime.now()
            cv2.imwrite("Unauthorised/"+str(CT.strftime("%Y-%m-%d-%H-%M-%S"))+".jpg",gray[y:y+h,x:x+w]) #add picture of faces in Folder Unauthorised
            cv2.imwrite("UnPicture/"+str(CT.strftime("%Y-%m-%d-%H-%M-%S"))+".jpg",img2) # add full size image in folder UnPicture
    cv2.imshow('Face',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
