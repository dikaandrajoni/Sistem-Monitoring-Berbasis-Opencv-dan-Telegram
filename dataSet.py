#Import Libary
import cv2
from imutils.video import VideoStream
import imutils
import os

#Memanggil kamera
cap = cv2.VideoCapture(0)
cap.set(3, 640) #Mengatur lebar frame
cap.set(4, 480) #Mengatur tinggi frame

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n Masukkan ID ==> ')

print("\n [INFO] Initialisasi data. Silahkan menghadap ke arah kamera untuk pengambilan data wajah...")
#Initialisasi jumlah sample wajah
count = 0

while (True):
    ret, img = cap.read()
    img = cv2.flip(img, -1) #Flip video image vertical
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1
        #Simpan hasil capture gambar ke dalam folder dataset
        cv2.imwrite("dataset/User."+str(face_id)+'.'+str(count)+".jpg", gray[y:y+h,x:x+w])

    cv2.imshow('image',img)
    k = cv2.waitKey(100) & 0xff #Press 'ESC' untuk menutup video
    if k==27:
        break
    elif count >=30: # Ambil 30 sample
        break

print ("\n [INFO] Menutup program dataset")
cap.release()
cv2.destroyAllWindows()












