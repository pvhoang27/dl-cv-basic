import json

# --- Dữ liệu giả định (dùng lại từ trên) ---
image_filename = "frames2/frame_001.jpg"
image_width = 1920
image_height = 1080
output_filename = "annotations.json"

annotations_pixel = [
    {"class_id": 0, "label": "xe", "bbox": [100, 200, 500, 450]},
    {"class_id": 1, "label": "người", "bbox": [650, 300, 750, 600]}
]

# --- Logic tạo cấu trúc và ghi file JSON ---

# Tạo một dictionary để chứa toàn bộ thông tin
data_for_json = {
    "image_info": {
        "filename": image_filename,
        "width": image_width,
        "height": image_height
    },
    "annotations": annotations_pixel
}

# Mở file và ghi dữ liệu dictionary vào
# indent=4 giúp file JSON có định dạng đẹp, dễ đọc
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(data_for_json, f, ensure_ascii=False, indent=4)

print(f"Đã tạo file annotation JSON tại: {output_filename}")