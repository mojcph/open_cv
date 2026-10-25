from picamera2 import Picamera2
import cv2
import threading

WIDTH, HEIGHT = 640, 480

frames = {"cam1": None, "cam2": None}
lock = threading.Lock()


def capture_camera(name, camera_num):
    camera = Picamera2(camera_num)

    config = camera.create_video_configuration(
        main={
            "size": (WIDTH, HEIGHT),
            "format": "RGB888"
        },
        buffer_count=4
    )

    camera.configure(config)
    camera.start()

    print(f"{name}: started camera {camera_num}")

    try:
        while True:
            # Picamera2 returns a NumPy image array.
            rgb = camera.capture_array()

            # Convert RGB to BGR for normal OpenCV processing.
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cv2.putText(
                frame, name, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2
            )

            with lock:
                frames[name] = frame
    finally:
        camera.stop()
        camera.close()


if __name__ == "__main__":
    threading.Thread(
        target=capture_camera,
        args=("cam1", 0),
        daemon=True
    ).start()

    threading.Thread(
        target=capture_camera,
        args=("cam2", 1),
        daemon=True
    ).start()

    # http://localhost:5000 
    app.run(
        host="127.0.0.1",
        port=5000,
        threaded=True,
        debug=False,
        use_reloader=False
    )
