import cv2

image = cv2.imread("du-lich.jpg")

image[:,:,1] = 0
image[:,:,2] = 0

cv2.imshow("Image", image)

cv2.waitKey(0)