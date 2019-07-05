import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path='Dataset'


def getImageWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[1].split('_')[1].split('-')[0])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.waitKey(10)
    return IDs, faces

IDs,faces=getImageWithID(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
