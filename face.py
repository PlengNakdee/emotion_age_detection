import numpy as np
import cv2
import keras
import tensorflow as tf
from keras.models import load_model

model = tf.keras.models.load_model('emotion_2.h5')
gender_model = tf.keras.models.load_model('gender_1.h5')
age_model = tf.keras.models.load_model('age_2.h5')

    # prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

    # dictionary for classes/labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
gender_classes = {0:'Male', 1:'Female'}
age_classes = {0:'1-10', 1:'11-20', 2:'21-30', 3:'31-40', 4:'41-50', 5:'51-60', 6:'61-70', 7:'71-80', 8:'81-90', 9:'91+'}

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray_2 = roi_gray.copy()

        #emotion input
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cropped_img = cropped_img.astype(np.float16)
        emo_pred = model.predict(cropped_img)
        emo_class = int(np.argmax(emo_pred))

        #gender input
        convert = cv2.cvtColor(roi_gray_2, cv2.COLOR_GRAY2RGB)
        cropped = np.expand_dims(cv2.resize(convert, (150, 150)), 0)
        cropped = cropped.astype(np.float16)
        cropped_2 = cropped.copy()
        gender_pred = gender_model.predict(cropped)
        gender_class = int(np.argmax(gender_pred))

        #age input
        age_pred = age_model.predict(cropped_2)
        age_class = int(np.argmax(age_pred))

        cv2.putText(frame, emotion_dict[emo_class], (x+20, y-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, gender_classes[gender_class], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, age_classes[age_class], (x+20, y-140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #print('object:', obj_detect, 'emotion:', emo_class, 'gender:', gender_class, 'age:', age_class)
        
    cv2.imshow('Video', cv2.resize(frame,(640, 480),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
