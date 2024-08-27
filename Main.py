import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime

    ########Initialize Firebase#######
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancesystem-32771-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendancesystem-32771.appspot.com"
})

bucket = storage.bucket()

#WEBCAMERA
webcam = cv2.VideoCapture(0)

imgBg = cv2.imread('bgImg/background.png')

ModePath = 'templates'
modePathList = os.listdir(ModePath)
imgModeList = [cv2.imread(os.path.join(ModePath, path)) for path in modePathList]


print("Loading Encode File ...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

modeNo = 0
counter = 0
id = -1
imgStudent = []

while True:
    success, frame = webcam.read()
    frame = cv2.flip(frame,1)

    Sframe = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    Sframe = cv2.cvtColor(Sframe, cv2.COLOR_BGR2RGB)

    ##### Face recognition #####
    faceCurFrame = face_recognition.face_locations(Sframe)
    encodeCurFrame = face_recognition.face_encodings(Sframe, faceCurFrame)

    ##### Update background image with webcam feed #####
    imgBg[162:162 + 480, 55:55 + 640] = frame
    imgBg[70:70 + 584, 830:830 + 368] = imgModeList[modeNo]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBg = cvzone.cornerRect(imgBg, bbox, rt=0)
                id = studentIds[matchIndex]

                if counter == 0:
                    cvzone.putTextRect(imgBg, "Scanning....", (197,400), 3, 3, (0,255,0), (0,0,0), cv2.FONT_HERSHEY_PLAIN)
                    cv2.imshow("Face Attendance", imgBg)
                    cv2.waitKey(1)
                    counter = 1
                    modeNo = 1

        if counter != 0:
            if counter == 1:
                # taking student information from Firebase Realtime Database
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)

                
                # retrieve student image from Firebase Storage
                blob = bucket.get_blob(f'Images/{id}.jpg')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                # Update attendance data
                datetimeObject = datetime.strptime(studentInfo['LAST MARKED'], "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()

                if secondsElapsed > 30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['TOTAL ATTENDANCE'] += 1
                    ref.child('TOTAL ATTENDANCE').set(studentInfo['TOTAL ATTENDANCE'])
                    ref.child('LAST MARKED').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeNo = 3
                    counter = 0
                    imgBg[70:70 + 584, 830:830 + 368] = imgModeList[modeNo]

            if modeNo != 3:
                if 10 < counter < 20:
                    modeNo = 2

                imgBg[70:70 + 584, 830:830 + 368] = imgModeList[modeNo]

                if counter <= 10:
                    cv2.putText(imgBg, str(studentInfo['NAME']), (877,126), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 2)
                    cv2.putText(imgBg, "ID: " + str(studentInfo['ROLL NO']), (890,454), cv2.FONT_HERSHEY_PLAIN, 1.9, (255, 0, 255), 2)
                    cv2.putText(imgBg, "COURSE: " + str(studentInfo['COURSE']), (890,510), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 0, 255), 2)
                    cv2.putText(imgBg, "BRANCH: " + str(studentInfo['BRANCH']), (890,566), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 0, 255), 2)
                    cv2.putText(imgBg, "BATCH: " + str(studentInfo['BATCH']), (890,622), cv2.FONT_HERSHEY_PLAIN, 1.7, (255, 0, 255), 2)
                    imgBg[170:170 + 216, 909:909 + 216] = imgStudent

                
                if modeNo == 1:
                    cv2.putText(imgBg, studentInfo['NAME'], (55 + x1, 162 + y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255, 0), 2)
                    cv2.putText(imgBg, "LAST MARKED: " + studentInfo['LAST MARKED'], (62,200), cv2.FONT_HERSHEY_PLAIN, 2, (128,0,128), 3)

                counter += 1

                if counter >= 20:
                    counter = 0
                    modeNo = 0
                    studentInfo = []
                    imgStudent = []
                    imgBg[70:70 + 584, 830:830 + 368] = imgModeList[modeNo]
    else:
        modeNo = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
