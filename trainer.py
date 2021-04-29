import cv2
import numpy as np
from PIL import Image
import os

#Nama folder untuk menyimpan data sampel wajah
path = 'dataset'

recognizer=cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

def get_images_and_labels(path):
     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
     images = []
     labels = []
     for image_path in image_paths:
         # Baca gambar dan convert ke grayscale
         image_pil = Image.open(image_path).convert('L')
         # Convert gambar ke numpy array
         image = np.array(image_pil, 'uint8')
         # Ambil label dari gambar
         nbr = int(os.path.split(image_path)[-1].split(".")[1])
         
         print(nbr)
         # Deteksi wajah
         faces = detector.detectMultiScale(image)
         # Jika wajah terdeteksi append the face to images and the label to labels
         for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             labels.append(nbr)
             cv2.imshow("Menambahkan data wajah untuk training...", image[y: y + h, x: x + w])
             cv2.waitKey(10)
     # return the images list and labels list
     return images, labels

print("\n [INFO] Training gambar wajah. Silahkan tunggu beberapa saat...")
images,labels = get_images_and_labels(path)
cv2.imshow('test',images[0])
cv2.waitKey(1)

recognizer.train(images, np.array(labels))
#Menyimpan data pola wajah ke dalam data YAML
recognizer.save('trainer/trainer.yml')
cv2.destroyAllWindows()

print("\n [INFO] Training {0} wajah selesai. Menutup Program".format(len(np.unique(labels))))




























        
