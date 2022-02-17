import cv2

image = cv2.imread('resources/group_of_people_08.jpg')

detector = cv2.CascadeClassifier('cascades/haarcascade_fullbody.xml')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detections = detector.detectMultiScale(image_gray)
print('[INFO] People detected: ' + str(len(detections)))

for (x, y, w, h) in detections:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow('People', image)

cv2.waitKey(0)
cv2.destroyAllWindows()