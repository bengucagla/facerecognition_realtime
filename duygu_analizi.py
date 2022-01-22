import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
model = load_model("C:/Users/BENGÜ ÇAĞLA/Desktop/best_model.h5")
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")
cap = cv2.VideoCapture(0)
label=['Cagla']


while True:
    ret, test = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    grey=cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
    yuz_tanima = cascade.detectMultiScale(gray, 1.32, 5)
    for (x, y, w, h) in yuz_tanima:
        cv2.rectangle(test, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray[y:y + w, x:x + h]
        roi_grey=grey[y:w+h,x:y+h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)
        id_,conf=recognizer.predict(roi_grey)
        if conf>=80:
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=label[id_]
            cv2.putText(test,name,(x,y+20),font,1,(0,0,255),2)
        max_index = np.argmax(predictions[0])
        duygular = ('kizgin', 'igrenme', 'korku', 'mutlu', 'uzgun', 'sasirmis', 'dogal')
        predicted_emotion = duygular[max_index]
        cv2.putText(test, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,)
    yeniden_boyutlandirma = cv2.resize(test, (1000, 800))
    cv2.imshow('Duygu analizi ', yeniden_boyutlandirma)

    if cv2.waitKey(10) == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows
