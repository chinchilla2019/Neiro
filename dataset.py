import cv2

cap = cv2.VideoCapture(0)
face_id = 1
face_casscade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for count in range(30):
    img = cap.read()[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_casscade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
        cv2.imshow(str(count), img)
        cv2.moveWindow(str(count), 10, 10)
        cv2.waitKey(500)
        cv2.destroyWindow(str(count))
    
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()