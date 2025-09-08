import numpy as np
from matplotlib import pyplot as plt
import os

# 1. Đường dẫn thư mục chứa dữ liệu
data_dir = r"C:\Users\hoang\Desktop\DL-CV-basic\data"

# 2. Load dữ liệu từ file .npy
file_path = os.path.join(data_dir, "full_numpy_bitmap_bicycle.npy")
images = np.load(file_path).astype(np.float32)
# print("Shape:", images.shape)

# 3. Tách dữ liệu train và test
train_images = images[:-10]
test_images = images[-10:]

# 4. Tính ảnh trung bình từ tập train
avg_image = np.mean(train_images, axis=0).reshape(28, 28)
# print("Avg shape:", avg_image.shape)

# 5. Hiển thị ảnh trung bình
# plt.imshow(avg_image, cmap="gray")
# plt.title("Ảnh trung bình của Bicycle")
# plt.show()

# 6. Tính điểm tương đồng cho một ảnh test
index = 4
test_image = test_images[index]
# print("Test shape:", test_image.shape)
# score = np.dot(test_image, avg_image.flatten())
# print("Score:", score)

# 7. So sánh dot product giữa các categories
categories = ["bicycle", "apple", "banana"]
scores = []
avg_images = []

for c in categories:
    file_path = os.path.join(data_dir, f"full_numpy_bitmap_{c}.npy")
    images = np.load(file_path).astype(np.float32)
    avg_image = np.mean(images, axis=0)
    avg_images.append(avg_image.reshape(28, 28))
    dot_prod = test_image @ avg_image
    scores.append(dot_prod)

# print("Scores:", scores)


plt.figure(figsize=(10, 4))
for i in range(len(categories)):
    plt.subplot(2,5, i+1 )
    plt.imshow(avg_images[i])
    plt.title(categories[i])

plt.show()