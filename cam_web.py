import cv2
import threading
import time
from flask import Flask, Response, jsonify, render_template
from picamera2 import Picamera2

WIDTH = 640
HEIGHT = 480
JPEG_QUALITY = 80

app = Flask(__name__)

frames = {"cam1": None, "cam2": None}
status = {"cam1": "Starting", "cam2": "Starting"}
lock = threading.Lock()

'''
@brief This script captures video from two Raspberry Pi cameras 
using the Picamera2 library and streams the combined video feed 
via a Flask web server. It initializes each camera, configures 
them for video capture, and starts separate threads to continuously 
capture frames. The captured frames are converted from RGB to BGR 
format for OpenCV compatibility and stored in a shared dictionary. 
The Flask server provides endpoints to view the live video stream 
and check the status of each camera.
'''
def capture_camera(name, camera_num):
    camera = None

    try:
        camera = Picamera2(camera_num)
        config = camera.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "RGB888"},
            buffer_count=4,
        )
        camera.configure(config)

        with lock:
            status[name] = "Configured"

        camera.start()

        with lock:
            status[name] = "Running"

        print(f"{name}: camera {camera_num} started", flush=True)

        while True:
            rgb = camera.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            with lock:
                frames[name] = frame

    except Exception as exc:
        with lock:
            status[name] = f"ERROR: {exc}"
        print(f"{name}: {exc}", flush=True)

    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
            try:
                camera.close()
            except Exception:
                pass

'''
@brief Generates a continuous stream of JPEG-encoded frames
from the two cameras, concatenated side by side.
This function is used by the Flask route to provide a live video feed.
It checks if frames from both cameras are available, combines them,
and yields the resulting JPEG image.
'''
def generate_stream():
    while True:
        with lock:
            left = frames["cam1"].copy() if frames["cam1"] is not None else None
            right = frames["cam2"].copy() if frames["cam2"] is not None else None

        if left is None or right is None:
            time.sleep(0.05)
            continue

        combined = cv2.hconcat([left, right])
        ok, jpeg = cv2.imencode(
            ".jpg",
            combined,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )

        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
        else:
            time.sleep(0.05)

'''
@brief Flask route for the main page, displaying the live video feed
from both cameras side by side. It includes a link to check the status 
of each camera.
'''
@app.route("/")
def index():
    return render_template("index.html")

'''
@brief Flask route for the video stream. It returns a multipart response
that continuously sends JPEG-encoded frames from the two cameras.
This allows the client to display a live video feed in a web browser.
'''
@app.route("/video")
def video():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

'''
@brief Flask route for checking the status of each camera.
It returns a JSON response containing the current status of both cameras,
such as "Starting", "Configured", "Running", or any error messages.
'''
@app.route("/status")
def camera_status():
    with lock:
        return jsonify(status)

'''
@brief Main entry point of the script. It initializes and starts the camera threads.
'''
if __name__ == "__main__":
    for name, camera_num in [("cam1", 0), ("cam2", 1)]:
        thread = threading.Thread(
            target=capture_camera,
            args=(name, camera_num),
            daemon=True,
        )
        thread.start()

        for _ in range(100):
            with lock:
                current = status[name]
            if current != "Starting":
                break
            time.sleep(0.05)

        if current.startswith("ERROR"):
            print(f"Check {name} before continuing.", flush=True)

    app.run(
        host="127.0.0.1",
        port=5000,
        threaded=True,
        debug=False,
        use_reloader=False,
    )
