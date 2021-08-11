import cv2
cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    _,frame = cam.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey,1.1,5)
    for (x,y,h,w) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,256,0),3)
    cv2.imshow("Face Recognition",frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()