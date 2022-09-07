''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18

'''

import cv2
import os
# import face_training 

cam = cv2.VideoCapture(0)
# cam.set(3, 640) # set video width
# cam.set(4, 480) # set video height
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

face_detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
eyeCascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades + 'haarcascade_eye.xml'))

# For each person, enter one numeric face id
face_id = input('\nEmployee ID: ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        count += 1
        # check if user has his eyes opened
        eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor= 1.5, minNeighbors=10, minSize=(5, 5),)

        # Save the captured image into the datasets folder if verified that it is a face
        if len(eyes) > 0:
            cv2.imwrite("dataset/hs_" + str(face_id) + '_' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 300: # Take 300 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
