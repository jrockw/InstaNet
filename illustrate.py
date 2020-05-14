from mtcnn import MTCNN
import cv2
photo = input("What photo would you like processed?")
img = cv2.imread(photo)
detector = MTCNN()
a = detector.detect_faces(img)
for i in a:
    i = i['box']
    cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,0,255),2)
cv2.imwrite('ellenFACE2.png',img)
cv2.imshow("RESULT", img)
cv2.waitKey(0)
