import cv2
import os

video_path = 'video/Inside these walls lives the softest Christmas magic🎄✨ #fyp #cozyvibes #foryou #cozyroom #fypシ .mp4'
out_dir = 'frames'
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{out_dir}/frame_{count:05d}.jpg", frame)
    count += 1

cap.release()
print("Đã tách xong", count, "frame")
