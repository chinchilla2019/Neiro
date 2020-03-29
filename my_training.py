import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)
        # img = Image.open(imagePath).convert('L')  # PIL
        img_numpy = np.array(img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_cascade.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer.yml')
