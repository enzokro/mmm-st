import cv2

def test_video_capture(device_path):
    cap = cv2.VideoCapture(device_path)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("Successfully captured a frame")
            #cv2.imshow("Test Frame", frame)
            #cv2.waitKey(0)  # Press any key to close the window
            #cv2.destroyAllWindows()
        else:
            print("Failed to capture a frame")
        cap.release()
    else:
        print("Failed to open video device")

# Replace 'device_path' with the actual device path, e.g., '/dev/video0' or '/dev/video1'
test_video_capture('/dev/video0')  # Make sure to try '/dev/video1' if '/dev/video0' does not work

