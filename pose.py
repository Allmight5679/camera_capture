#just to add the camera access
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

print("Camera working:", ret)

cap.release()