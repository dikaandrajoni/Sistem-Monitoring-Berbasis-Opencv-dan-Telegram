import telepot
from telepot.loop import MessageLoop
import cv2,os
import numpy as np
from PIL import Image 
import pickle
import time
from datetime import datetime

now = datetime.now()
bot = telepot.Bot('1622048284:AAFILnGIR8s17YSSkwI87wjGp_Pxdb5c9Lw')
id = '648642884'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataset'
count=0
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
print("Start...")
bot.sendMessage(id, "{}\n[INFO] Start sistem monitoring...".format(now))
while True:
    ret, im =cam.read()
    im = cv2.flip(im, -1)
    #im = cv2.blur(im,(5,5))
    im = cv2.GaussianBlur(im,(5,5),0)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-20,y-20),(x+w+20,y+h+20),(225,0,0),2)
        print(nbr_predicted, conf)
        if conf < 80 :
            if(nbr_predicted==1):
                nbr_predicted='Dika'
            elif(nbr_predicted==2):
                nbr_predicted='Andra'
            elif(nbr_predicted==3):
                nbr_predicted='Joni'
            else:
                nbr_predicted='Unknown'
                
        # Jika wajah terdeteksi Unknown kirim pesan dan data wajah ke bot telegram
        if nbr_predicted=='Unknown':
            count += 1
            name = "capture/unknown{}.jpg".format(count)
            cv2.imwrite(name, im)
            photo = open(name, 'rb')
            bot.sendMessage (id, str("\n [INFO] Orang tidak dikenal terdeteksi!!!"))
            time.sleep(5)
            bot.sendPhoto(id, photo)
            
        n = int(conf)
        cv2.putText(im,str(nbr_predicted)+"--"+str(n)+"%", (x,y+h),font, 0.5, (255,0,0),1) #Draw the text
        cv2.imshow('im',im)
    if cv2.waitKey(10) & 0xff==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
    









            
       

















































