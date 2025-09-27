import os

# --- Dữ liệu giả định ---
image_width = 1920
image_height = 1080
output_filename = "frame_001.txt"

# Danh sách các đối tượng đã được gán nhãn [class_id, xmin, ymin, xmax, ymax]
annotations = [
    [0, 100, 200, 500, 450], # [xe]
    [1, 650, 300, 750, 600]  # [người]
]

# --- Logic chuyển đổi và ghi file ---

# Mở file để ghi, 'w' là chế độ ghi đè
with open(output_filename, 'w') as f:
    for anno in annotations:
        class_id, xmin, ymin, xmax, ymax = anno

        # Tính toán tọa độ tâm và kích thước của bounding box (pixel)
        box_width = xmax - xmin
        box_height = ymax - ymin
        center_x = xmin + box_width / 2
        center_y = ymin + box_height / 2

        # Chuẩn hóa tọa độ về khoảng [0, 1]
        norm_center_x = center_x / image_width
        norm_center_y = center_y / image_height
        norm_width = box_width / image_width
        norm_height = box_height / image_height

        # Tạo chuỗi định dạng YOLO: "class_id center_x center_y width height"
        yolo_string = f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n"

        # Ghi vào file
        f.write(yolo_string)

print(f"Đã tạo file annotation YOLO tại: {output_filename}")