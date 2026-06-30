"""Lightweight MJPEG-over-HTTP streamer for headless display.

Optimisations (v2):
  - Async JPEG encoding in a background thread (non-blocking ``update_frame``).
  - ``TCP_NODELAY`` on the server socket to disable Nagle's algorithm.
  - ``cv2.IMWRITE_JPEG_OPTIMIZE`` for smaller JPEGs at equal quality.
  - Configurable *stream_skip* to throttle encoding rate independently.
  - Pre-built MJPEG boundary strings to avoid per-frame allocations.
  - Aggressive ``wfile.flush()`` after each frame for lower latency.

Usage::

    streamer = MJPEGStreamer(port=11451, quality=50, stream_skip=2)
    streamer.start()

    while processing:
        streamer.update_frame(annotated_frame)   # non-blocking!

    streamer.stop()
"""

import http.server
import socket
import threading
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Pre-computed constants for the MJPEG boundary to avoid per-frame allocations
# ---------------------------------------------------------------------------
_BOUNDARY = b"--frame\r\n"
_CT_JPEG = b"Content-Type: image/jpeg\r\n"
_CL_PREFIX = b"Content-Length: "
_CRLF2 = b"\r\n\r\n"
_CRLF1 = b"\r\n"


class _StreamHandler(http.server.BaseHTTPRequestHandler):
    """Serve ``/`` as MJPEG stream and ``/snapshot`` as a still JPEG."""

    # Class-level reference to the streamer instance (set by the server thread).
    streamer: "MJPEGStreamer | None" = None

    # ------------------------------------------------------------------
    def do_GET(self):
        if self.path == "/":
            self._serve_mjpeg()
        elif self.path == "/snapshot":
            self._serve_snapshot()
        elif self.path == "/health":
            self._serve_text(200, "OK")
        else:
            self._serve_text(404, "Not Found")

    # ------------------------------------------------------------------
    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=--frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")       # disable nginx proxy buffering
        self.end_headers()

        # Disable Nagle on this connection's socket for lower latency
        try:
            self.connection.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        s = self.streamer
        last_id = -1
        # Pre-allocate a reusable bytearray for the Content-Length header line
        try:
            while s and s._running:
                jpeg, new_id = s.wait_for_new_frame(last_id, timeout=1.0)
                if jpeg is None:
                    continue
                last_id = new_id

                # –– write MJPEG multipart frame ––
                wf = self.wfile
                wf.write(_BOUNDARY)                          # --frame\r\n
                wf.write(_CT_JPEG)                           # Content-Type: image/jpeg\r\n
                wf.write(_CL_PREFIX)                         # Content-Length:
                wf.write(str(len(jpeg)).encode())            # <length>
                wf.write(_CRLF2)                             # \r\n\r\n
                wf.write(jpeg)                               # <jpeg bytes>
                wf.write(_CRLF1)                             # \r\n
                wf.flush()                                   # push to client immediately
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    # ------------------------------------------------------------------
    def _serve_snapshot(self):
        jpeg, _ = self.streamer.get_latest_frame(timeout=2.0)
        if jpeg is None:
            self._serve_text(204, "")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(jpeg)
        self.wfile.flush()

    # ------------------------------------------------------------------
    def _serve_text(self, code, text):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        payload = text.encode("utf-8") if isinstance(text, str) else text
        self.wfile.write(payload)
        self.wfile.flush()

    # ------------------------------------------------------------------
    def log_message(self, fmt, *args):
        pass  # suppress access log noise


# ======================================================================
class MJPEGStreamer:
    """Runs a lightweight HTTP server that streams the latest frame as MJPEG.

    **New in v2** – asynchronous JPEG encoding:

    ``update_frame()`` only copies the raw NumPy array and returns
    immediately.  A dedicated encoder thread performs the actual
    ``cv2.imencode`` call in the background, so the main processing
    loop is never blocked by JPEG compression.

    Parameters
    ----------
    port : int
        HTTP listen port (default 11451).
    quality : int
        JPEG quality 1–100 (default 80).  Lower = smaller files & less
        encoding time but grainier images.
    stream_skip : int
        Only encode & push every N-th call to ``update_frame`` (default 1
        = every frame).  Useful for reducing CPU load independently of the
        AI ``--frame-skip``.
    bind : str
        Bind address (default ``0.0.0.0``).
    resize : tuple | None
        ``(width, height)`` to downscale frames before encoding, or
        ``None`` to keep original size.
    """

    def __init__(self, port: int = 11451, quality: int = 80,
                 stream_skip: int = 1,
                 bind: str = "0.0.0.0", resize: tuple | None = None):
        self._port = port
        self._quality = max(1, min(100, int(quality)))
        self._stream_skip = max(1, int(stream_skip))
        self._bind = bind
        self._resize = resize  # e.g. (1024, 576) or None to keep original
        self._running = False
        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._enc_thread: threading.Thread | None = None

        # ── consumer side: condition + latest JPEG ──
        self._cond = threading.Condition()
        self._latest_jpeg: bytes | None = None
        self._frame_id: int = 0
        self._last_update = 0.0

        # ── producer side: raw frame → encoder thread ──
        self._raw_cond = threading.Condition()
        self._raw_frame: np.ndarray | None = None
        self._raw_frame_id: int = 0          # monotonically increasing
        self._update_count: int = 0           # for stream_skip counting

    # ------------------------------------------------------------------
    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self._bind}:{self._port}"

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    def start(self) -> "MJPEGStreamer":
        """Begin serving the MJPEG stream in a background thread."""
        if self._running:
            return self

        handler = type("_BoundHandler", (_StreamHandler,), {"streamer": self})

        for _ in range(5):
            try:
                self._server = http.server.ThreadingHTTPServer(
                    (self._bind, self._port), handler)
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"Could not bind to {self._bind}:{self._port}")

        # ── Disable Nagle on the *listening* socket so every accepted
        #    connection inherits TCP_NODELAY automatically ──
        try:
            self._server.socket.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        self._running = True

        # –– background encoder thread ––
        self._enc_thread = threading.Thread(target=self._encoder_loop, daemon=True)
        self._enc_thread.start()

        # –– HTTP server thread ––
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        print(f"MJPEG streamer started at {self.url}  "
              f"(quality={self._quality}, stream_skip={self._stream_skip})")
        return self

    # ------------------------------------------------------------------
    def stop(self):
        """Shut down the HTTP server and encoder thread gracefully."""
        self._running = False
        # unblock encoder thread
        with self._raw_cond:
            self._raw_cond.notify_all()
        # unblock HTTP consumer threads
        with self._cond:
            self._cond.notify_all()

        srv = self._server
        if srv is not None:
            try:
                srv.shutdown()
            except Exception:
                pass
        for t in (self._thread, self._enc_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        print("MJPEG streamer stopped.")

    # ------------------------------------------------------------------
    #  PRODUCER API  (called from main processing thread)
    # ------------------------------------------------------------------

    def update_frame(self, frame: np.ndarray):
        """Submit a new frame for asynchronous encoding (non-blocking).

        The frame is **copied** immediately, so the caller can reuse the
        array.  Actual JPEG compression happens in a background thread.
        When *stream_skip* > 1, only every N-th call actually submits.
        """
        if not self._running:
            return

        # ── stream-level frame skipping (independent of AI frame-skip) ──
        self._update_count += 1
        if self._stream_skip > 1 and (self._update_count % self._stream_skip != 1):
            return

        # ── downscale if configured ──
        if self._resize is not None:
            frame = cv2.resize(frame, self._resize, interpolation=cv2.INTER_NEAREST)

        # ── copy so caller can reuse the array ──
        frame_copy = frame.copy()

        with self._raw_cond:
            self._raw_frame = frame_copy
            self._raw_frame_id += 1
            self._last_update = time.time()
            self._raw_cond.notify_all()       # wake up encoder thread

    # ------------------------------------------------------------------
    #  CONSUMER API  (called from HTTP handler threads)
    # ------------------------------------------------------------------

    def get_latest_frame(self, timeout: float = 1.0) -> tuple[bytes | None, int]:
        """Return ``(jpeg_bytes, frame_id)`` of the current latest frame.

        Returns ``(None, 0)`` if no frame has been published yet.
        """
        deadline = time.time() + timeout
        with self._cond:
            while self._latest_jpeg is None and time.time() < deadline:
                if not self._cond.wait(max(deadline - time.time(), 0.001)):
                    break
            return (self._latest_jpeg, self._frame_id)

    # ------------------------------------------------------------------
    def wait_for_new_frame(self, last_id: int, timeout: float = 1.0
                           ) -> tuple[bytes | None, int]:
        """Block until a frame with ``id > last_id`` is available.

        Returns ``(jpeg_bytes, frame_id)``.  Returns ``(None, last_id)``
        on timeout or if the streamer has been stopped.
        """
        deadline = time.time() + timeout
        with self._cond:
            while self._running and self._frame_id <= last_id:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return (None, last_id)
                self._cond.wait(remaining)
            if self._frame_id > last_id:
                return (self._latest_jpeg, self._frame_id)
            return (None, last_id)

    # ------------------------------------------------------------------
    #  INTERNAL
    # ------------------------------------------------------------------

    def _encoder_loop(self):
        """Background thread: encode raw frames → JPEG.

        Waits efficiently on ``_raw_cond`` and only encodes when a new
        raw frame has been submitted.  Skips duplicate work when multiple
        frames arrive before encoding finishes (only encodes the latest).
        """
        last_encoded_id = -1
        while self._running:
            # ── wait for a new raw frame ──
            with self._raw_cond:
                while self._raw_frame_id == last_encoded_id and self._running:
                    self._raw_cond.wait(timeout=0.5)
                if not self._running:
                    break
                # grab latest frame under the lock, then release for encoding
                raw = self._raw_frame
                fid = self._raw_frame_id

            # ── encode outside the lock (may take 10–30 ms) ──
            if raw is None:
                last_encoded_id = fid
                continue

            success, jpeg = cv2.imencode(".jpg", raw, [
                cv2.IMWRITE_JPEG_QUALITY, self._quality,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,        # better Huffman tables
            ])
            if not success:
                last_encoded_id = fid
                continue

            jpeg_bytes = jpeg.tobytes()

            # ── publish to consumers ──
            with self._cond:
                self._latest_jpeg = jpeg_bytes
                self._frame_id = fid
                self._cond.notify_all()

            last_encoded_id = fid

    # ------------------------------------------------------------------
    def _serve(self):
        """Internal: main loop of the HTTP server."""
        try:
            self._server.serve_forever(poll_interval=0.1)
        except Exception:
            pass
        finally:
            self._running = False

    # ------------------------------------------------------------------
    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
