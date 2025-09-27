import cv2

# Đường dẫn file đã được sửa bằng cách thêm 'r'
video_path = r'C:\Users\hoang\Desktop\DL-CV-basic\video\Inside these walls lives the softest Christmas magic🎄✨ #fyp #cozyvibes #foryou #cozyroom #fypシ .mp4'

# Tạo video capture object
vid_capture = cv2.VideoCapture(video_path)

if not vid_capture.isOpened():
    print("Error opening the video file")
else:
    # Lấy FPS (frame per second)
    fps = vid_capture.get(cv2.CAP_PROP_FPS)
    print('Frames per second :', fps, 'FPS')

    # Lấy tổng số frame
    frame_count = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Frame count :', frame_count)

# Đọc và hiển thị video
while vid_capture.isOpened():
    ret, frame = vid_capture.read()
    if ret:
        cv2.imshow('Frame', frame)

        # Chờ 20ms, nhấn 'q' để thoát
        # Dùng 1 giá trị lớn hơn 0, ví dụ 20, để video không chạy quá nhanh
        key = cv2.waitKey(20)
        if key == ord('q'):
            break
    else:
        break

# Giải phóng bộ nhớ
vid_capture.release()
cv2.destroyAllWindows()