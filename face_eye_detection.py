import cv2

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')


img = cv2.imread('./dataset_new/train/ny/0_613.jpg')


r = 500.0 / img.shape[1]
dim = (500, int(img.shape[0] * r))


resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


cv2.imshow('image', resized)
cv2.waitKey(0)


grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

cv2.imshow('image', grey)
cv2.waitKey(0)

faces = face_cascade.detectMultiScale(grey, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_grey = grey[y:y+h, x:x+w]
    roi_color = resized[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_grey)
for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)


cv2.imshow('img', resized)
cv2.waitKey(0)
