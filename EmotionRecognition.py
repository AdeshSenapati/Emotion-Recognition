import cv2
import mediapipe as mp
import csv
import numpy as np
import pickle
import pandas as pd
import time

with open('Emotion_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture('videos/2.mp4')
pTime = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)  # static mode is false as detection is more heavier than
# tracking so it will first detect and then it will track if its true it will do both together
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

while True:
    success, img = cap.read()
    tt = []
    class_name = 'Neutral'
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                #x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z)
                tt.append(id)
                face_row = list(
                    np.array([[lm.x, lm.y, lm.z] for landmark in tt]).flatten())
    try:
        X = pd.DataFrame([face_row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]
        print(body_language_class, body_language_prob)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.putText(img, body_language_class, (300, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, str(round(body_language_prob[np.argmax(body_language_prob)])*100), (450, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    except:
        pass

    cv2.imshow("Image", img)
    cv2.waitKey(1)