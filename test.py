import cv2
import torch

image = cv2.imread("du-lich.jpg")
cv2.imshow("image", image)
image = torch.from_numpy(image)

print("Image's shape: {}".format(image.shape))
print("Image's number of dimensions: {}".format(image.ndim))

cv2.waitKey(0)