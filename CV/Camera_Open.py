import cv2

cap = cv2.VideoCapture(1)

while(True):
    ret, frame = cap.read()
    
    cv2.imshow('window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#do some ops

cap.release()
cv2.imshow("output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()