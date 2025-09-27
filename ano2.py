import cv2
import os

# --- Cấu hình ---
video_path = 'video/Inside these walls lives the softest Christmas magic🎄✨ #fyp #cozyvibes #foryou #cozyroom #fypシ .mp4'  # Đường dẫn đến video của bạn
output_folder = 'frames_2'      # Tên thư mục để lưu frames

# --- Logic ---

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Đã tạo thư mục: {output_folder}")

# Mở video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Lỗi: Không thể mở file video tại '{video_path}'")
    exit()

# Đọc video từng frame
frame_count = 0
while True:
    # ret là True nếu đọc frame thành công, ngược lại là False
    ret, frame = cap.read()

    if not ret:
        print("Đã xử lý xong tất cả các frame hoặc có lỗi.")
        break

    # Tạo tên file cho frame
    # Ví dụ: frames/frame_00001.jpg
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")

    # Lưu frame thành file ảnh
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

    # In ra tiến trình (có thể xóa dòng này nếu không cần)
    if frame_count % 100 == 0:
        print(f"Đã lưu {frame_count} frames...")

# Giải phóng tài nguyên
cap.release()
print(f"Hoàn tất! Đã lưu tổng cộng {frame_count} frames vào thư mục '{output_folder}'.")