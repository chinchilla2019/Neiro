import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

font = cv2.FONT_HERSHEY_SIMPLEX
id_index = 0
names = ['Unknown', 'Kate']

while True:
    img = cap.read()[1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id_index = recognizer.predict(gray[y:y + h, x:x + w])[0]
        id_name = names[id_index]
        cv2.putText(img, str(id_name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

    cv2.imshow('OpenCV Detection', img)
    key = cv2.waitKey(100)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
