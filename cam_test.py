import cv2
from picamera2 import Picamera2

# 1. Initialize the Raspberry Pi Camera via Picamera2
picam = Picamera2()

# 2. Configure camera settings (Resolution and Format)
# RGB888 outputs array formats that OpenCV natively understands
picam.configure(picam.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
))

# 3. Start the camera sensor stream
picam.start()

print("Camera started. Press 'q' in the window to quit.")

try:
    while True:
        # 4. Capture a live frame directly as a NumPy array
        frame = picam.capture_array()
        
        # 5. Picamera2 captures in RGB, but OpenCV defaults to BGR.
        # We must convert the color space so colors display correctly.
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 6. Display the live feed in an OpenCV GUI window
        cv2.imshow("Raspberry Pi 5 Camera", bgr_frame)
        
        # Break the loop immediately if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 7. Clean up and release hardware resources when exiting
    picam.stop()
    cv2.destroyAllWindows()
