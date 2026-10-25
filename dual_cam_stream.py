
import cv2
import threading
import time
from flask import Flask, Response, jsonify

from picamera2 import Picamera2

WIDTH = 640
HEIGHT = 480
JPEG_QUALITY = 80

app = Flask(__name__)

frames = {"cam1": None, "cam2": None}
status = {"cam1": "Starting", "cam2": "Starting"}
lock = threading.Lock()
cameras = []


def capture_camera(name, camera_num):
    camera = None

    try:
        # Create and configure each camera before starting it.
        camera = Picamera2(camera_num)

        config = camera.create_video_configuration(
            main={
                "size": (WIDTH, HEIGHT),
                "format": "RGB888",
            },
            buffer_count=4,
        )
        camera.configure(config)

        # Publish status before starting.
        with lock:
            status[name] = "Configured"

        camera.start()
        cameras.append(camera)

        with lock:
            status[name] = "Running"

        print(f"{name}: camera {camera_num} started", flush=True)

        while True:
            rgb = camera.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cv2.putText(
                frame,
                name,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

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


def generate_mjpeg():
    while True:
        with lock:
            left = frames["cam1"]
            right = frames["cam2"]

            left = left.copy() if left is not None else None
            right = right.copy() if right is not None else None

        if left is None or right is None:
            time.sleep(0.05)
            continue

        combined = cv2.hconcat([left, right])
        ok, encoded = cv2.imencode(
            ".jpg",
            combined,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )

        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )


@app.route("/")
def index():
    return """
    <!doctype html>
    <html>
    <head>
      <title>Pi 5 Dual Camera Preview</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        body { background: #181818; color: white;
               font: 16px sans-serif; margin: 20px; }
        img { width: 100%; max-width: 1280px; }
      </style>
    </head>
    <body>
      <h2>Raspberry Pi 5 - Dual Camera Preview</h2>
      <p>Camera 0 on left; camera 1 on right.</p>
      <img src="/stream">
      <p><a style="color:#9cf" href="/status">Camera status</a></p>
    </body>
    </html>
    """


@app.route("/stream")
def stream():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def camera_status():
    with lock:
        return jsonify(status)


if __name__ == "__main__":
    # Configure/start camera 0 first, then camera 1.
    # Keep Flask on localhost for VS Code SSH forwarding.
    for name, num in [("cam1", 0), ("cam2", 1)]:
        thread = threading.Thread(
            target=capture_camera,
            args=(name, num),
            daemon=True,
        )
        thread.start()

        # Give each camera time to initialize before starting the next.
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
