import numpy as np
import face_recognition
import cv2

#train Image
imgRazeen = face_recognition.load_image_file('images/razeen.jpg')
imgRazeen = cv2.cvtColor(imgRazeen,cv2.COLOR_BGR2RGB)

#Test Image
imgRazeenTest = face_recognition.load_image_file('razeen_2.jpg')
imgRazeenTest = cv2.cvtColor(imgRazeenTest,cv2.COLOR_BGR2RGB)

#Finding the faces & encodings in a Image

faceLocation = face_recognition.face_locations(imgRazeen)[0]
encode_imgRazeen = face_recognition.face_encodings(imgRazeen)[0]


faceLocationTest = face_recognition.face_locations(imgRazeenTest)[0]
encode_imgRazeenTest = face_recognition.face_encodings(imgRazeenTest)[0]


#Draw Rectangle around the face
cv2.rectangle(imgRazeen,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]),(255,0,255),2)
cv2.rectangle(imgRazeenTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encode_imgRazeen],encode_imgRazeenTest)
facedis = face_recognition.face_distance([encode_imgRazeen],encode_imgRazeenTest)

cv2.putText(imgRazeenTest,f'{results}{round(facedis[0],2)}',(150,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
print(results,facedis)


cv2.imshow("RazeenTrain",imgRazeen)
cv2.imshow("RazeenTest",imgRazeenTest)
cv2.waitKey(0)




