import numpy as np
import cv2
from my_dataset import MyDataset  # Đảm bảo bạn đã định nghĩa class MyDataset

if __name__ == '__main__':
    # Khởi tạo dataset
    dataset = MyDataset(root="data/cifar-10-batches-py", train=True)

    # Lấy một ảnh và nhãn (label)
    image, label = dataset.__getitem__(234)

    # Reshape ảnh từ vector 1D -> 3D (3 kênh màu, 32x32)
    image = np.reshape(image, (3, 32, 32))

    # Đảo thứ tự trục từ (C, H, W) -> (H, W, C) để hiển thị bằng OpenCV
    image = np.transpose(image, (1, 2, 0))

    # Kiểm tra dtype và label
    print("Image dtype:", image.dtype)
    print("Label:", label)

    # Hiển thị ảnh
    cv2.imshow("image", cv2.resize(image, (320, 320)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
