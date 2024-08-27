from builtins import print 
import cv2
import os
import face_recognition
import pickle

#this part for storing student's images in firebase storage
#START
import firebase_admin 
from firebase_admin import credentials 
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://faceattendancesystem-32771-default-rtdb.firebaseio.com/",
    'storageBucket':"faceattendancesystem-32771.appspot.com"
})
#end


# Fetching path of students images 
imgPath = 'Images'
imgPathList = os.listdir(imgPath)

# Importing students images and their IDs to a list
imgList = []
studentIDs = []
for path in imgPathList:
    img = cv2.imread(os.path.join(imgPath, path))       #here we are getting and storing student's images in stendentlist one by one
    imgList.append(img)
    studentIDs.append(os.path.splitext(path)[0])

    fileName=f'{imgPath}/{path}'             #here we are uploading images to firebase storage
    bucket=storage.bucket()
    blob=bucket.blob(fileName)
    blob.upload_from_filename(fileName)

    

#encoding
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
           img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
           encode = face_recognition.face_encodings(img)[0]
           encodeList.append(encode)
        
            

    return encodeList

print("Encoding starts...")
encodeListKnown = findEncodings(imgList)  # Function call
encodeListKnownWithIDs = [encodeListKnown,studentIDs]
print("Encoding complete")


file=open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIDs,file)
file.close()
print("encoding with id is saved in pickle file")
