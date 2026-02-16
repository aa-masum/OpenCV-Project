import cv2
import numpy as np

# Load webcam
cap = cv2.VideoCapture(0)

# Load pre-trained face classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=80, varThreshold=50, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))

    # ---------- Motion Detection ----------
    fgmask = fgbg.apply(frame)
    # Remove noise
    fgmask = cv2.medianBlur(fgmask, 5)
    # Find contours of moving objects
    contours, _ = cv2.findContours(
        fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # filter small movements
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Motion Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------- Face Detection ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.4, minNeighbors=3)
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
        cv2.putText(frame, "Person Detected", (fx, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show video
    cv2.imshow("Smart Motion Detector", frame)
    cv2.imshow("Motion Mask", fgmask)
    cv2.imwrite("motion_detected.jpg", frame)

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
