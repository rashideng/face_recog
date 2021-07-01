import numpy as np
import face_recognition
import cv2
import os

path = 'images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curIMg = cv2.imread(f'{path}/{cl}')
    images.append(curIMg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnown = findEncodings(images)
print("Encode Finshed")


cap = cv2.VideoCapture(1)
while True:
    success,img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeFaces,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFaces)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFaces)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)

        cv2.imshow('WebCam',img)
        cv2.waitKey(1)





























