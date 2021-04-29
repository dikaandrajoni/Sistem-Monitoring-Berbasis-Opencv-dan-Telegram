import os
import cv2
import telepot
import numpy as np
import urllib.request
import time
from datetime import datetime
from telepot.loop import MessageLoop

now = datetime.now()

print("\n [INFO] Start sistem monitoring...")

#Set bot token
telegram_bot = telepot.Bot('1622048284:AAFILnGIR8s17YSSkwI87wjGp_Pxdb5c9Lw')
#telegram_bot.sendMessage ('1622048284', str("[INFO] Start sistem monitoring..."))

#Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

#Data face trainer
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0

#names related to id: example ==> Alex: id=1, etc
names=['None','Dika','Alex','Ryan','W']

#Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) #Set video width
cam.set(4, 480) #Set video height

#Define min windows size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

def cekKoneksi():
    try:
        urllib.request.urlopen('https://www.google.com/',  timeout=1)
        return "Connect"
    
    except urllib.request.URLError as err:
        return "Disconnect, Silahkan periksa koneksi internet anda!"

def action(msg):
    chat_id = msg['chat']['id']
    command = msg['text']
    print ('Received: %s' % command)
    if command == '/hi':
        telegram_bot.sendMessage (chat_id, str("Hi! Dika Ganteng"))
    elif command == '/start':
        telegram_bot.sendMessage (chat_id, str("Your sistem ready!"))
    elif command == '/time':
        telegram_bot.sendMessage(chat_id, str(now.hour)+str(":")+str(now.minute))
    elif command == '/capture':
        timestamp = datetime.now()
        filename="capture/{}.jpg".format(timestamp)
        cv2.imwrite(filename, img)
        photo = open(filename,'rb')
        telegram_bot.sendPhoto (chat_id, photo = photo)
    else:
       telegram_bot.sendMessage (chat_id, str("Your Command Not Found!"))
     

while (True):
    ret, img = cam.read()
    img = cv2.flip(img, -1) #flip Vertical
    u_id = '1622048284'
    if cekKoneksi()=='Connect':
        MessageLoop(telegram_bot, action).run_as_thread()
    else:
        cv2.putText(img, str('Disconnect to intenet'),(10,20), font, 0.5,(0,0,255), 1)
        print(cekKoneksi())
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW),int(minH)),
        )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])

        print (conf)

        #Check if conf is less them 100 ==> "0" is perfect match
        if id == '1':
            user=names[id]
            conf = " {0}%".format(round(100-conf))
            if cekKoneksi()=='Connect':
                telegram_bot.sendMessage (u_id, str("Hi! {} Ganteng".format(user)))
            else:
                print(cekKoneksi())
               
        else:
            user="Unknown"
            conf = " {0}%".format(round(100-conf))
            timestamp = datetime.now()
            filename="capture/{}.jpg".format(timestamp)
            cv2.imwrite(filename, img)
            photo = open(filename,'rb')
            telegram_bot.sendMessage (u_id, str("Unknown!"))
            time.sleep(5)
            #Send foto to bot telegram
            #telegram_bot.send_photo(u_id, photo)
              

        cv2.putText(img, str(id), (x+10,y), font, 1,(255,255,255), 2)
        cv2.putText(img, str(conf), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('Monitoring', img)
    time.sleep(10)

    k = cv2.waitKey(10)&0xff
    if k == 27:
        break

print("\n [INFO] Menutup Program!")
cam.release()
cv2.destroyAllWindows()



















            
            
       

















































