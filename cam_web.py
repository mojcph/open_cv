import cv2
import threading
from flask import Flask, Response

app = Flask(__name__)

# Replace these with your existing camera frames.
# Your capture loop should continuously update these.
frames = {"cam1": None, "cam2": None}
lock = threading.Lock()


def generate_stream():
    while True:
        with lock:
            left = frames["cam1"]
            right = frames["cam2"]

            if left is None or right is None:
                continue

            # Scale frames to the same height before combining.
            height = 360
            left = cv2.resize(
                left, (640, height)
            )
            right = cv2.resize(
                right, (640, height)
            )

            combined = cv2.hconcat([left, right])
            ok, jpeg = cv2.imencode(".jpg", combined)

        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )


@app.route("/")
def index():
    return """
    <html>
      <head><title>Pi Camera Preview</title></head>
      <body>
        <h2>Raspberry Pi 5 - Dual Camera</h2>
        <img src="/video" style="width:100%;max-width:1280px">
      </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)