import cv2

cap = cv2.VideoCapture(0)  # Change the index to 0
cap.set(3, 800)
cap.set(4, 533)

imgBackground = cv2.imread('Resources/backaground.png')

while True:
    success, img = cap.read()

    imgBackground[ ] = img

    cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)

