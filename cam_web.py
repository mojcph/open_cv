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


@app.route("/")
def index():
    return """
    <html>
      <head><title>Pi Camera Preview</title></head>
      <body>
        <h2>Raspberry Pi 5 - Dual Camera</h2>
        <img src="/video" style="width:100%;max-width:1280px">
        <p><a href="/status">Camera status</a></p>
      </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def camera_status():
    with lock:
        return jsonify(status)


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
    