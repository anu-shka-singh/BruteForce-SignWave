import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("SignLanguageToText\Model\keras_model.h5", "SignLanguageToText\Model\labels.txt")
offset = 20
imgSize = 300


#folder = "Data/L"
counter = 0

labels = ["1","2","3","4","5","6","7","8","9","A","bad","C","good","Hello","how","L","O","okay","R","thank you","what","Y","you","I am"]


s = ""
num = ""
start_time = time.time()
stable_duration = 2  # duration for stable detections in seconds

# initialize the TTS engine
engine = pyttsx3.init()
# set the voice properties
newVoiceRate = 150 # set new rate
engine.setProperty('rate', newVoiceRate)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # set the voice as the first available voice

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
        
        """ if(len(num) == 0):
            
            num = num + str(index)
            
        elif(num[-1] != str(index)):
            
            num = num + str(index)
            s = s + labels[index] + " " """
            
        num_len = len(num)
        
        if num_len == 0 or (num[-1] != index and time.time() - start_time >= stable_duration):
            
            s = s + labels[index] + " " if num_len > 0 else s
            num = str(index)
            start_time = time.time()
            
        elif num[-1] != index and time.time() - start_time < stable_duration:
            
            num = num + str(index)

        #cv2.rectangle(imgOutput, (x - offset, y - offset),
        #              (x + w + offset, y + h + offset), (255, 0, 255), 4)

        #cv2.imshow("ImageCrop", imgCrop)
        #cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    # Display the string on the new window
    #cv2.putText(imgOutput, s, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #cv2.imshow("String", imgOutput)
    
    if cv2.waitKey(1) == ord('q'):
    	break

cap.release()
cv2.destroyAllWindows()

print(s)

# convert the sentence to speech and play it
engine.say(s)
engine.runAndWait()