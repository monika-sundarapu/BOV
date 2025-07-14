import cv2

def capture_image(filename="captured.jpg"):
    cap = cv2.VideoCapture(0)
    print("📸 Press SPACE to capture image")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Press SPACE to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite(filename, frame)
            print(f"✅ Image saved as {filename}")
            break

    cap.release()
    cv2.destroyAllWindows()
