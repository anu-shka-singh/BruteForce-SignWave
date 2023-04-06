import cv2
import numpy as np
import keras

cap = cv2.VideoCapture(0)

num_frames = 20

frames = []

labels = ["1","2","3","4","5","6","7","8","9","A","bad","C","good","Hello","how","L","O","okay","R","thank you","what","Y","you","I am"]


while True :
    success, img = cap.read()
    frame = cv2.resize(frame, (224,224))
    frame = frame.astype('float32')/255.0
    frames.append(frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or len(frames) >= num_frames:
        break
    
cap.release()
test_images = np.array(frames)
model = keras.models.load_model('/Users/anushka/Documents/BruteForce-SignWave/SignLanguageToText/Model/keras_model.h5')

test_loss, test_acc = model.evaluate(test_images, labels)
print('Test accuracy:', test_acc)

# Destroy all windows
cv2.destroyAllWindows()

