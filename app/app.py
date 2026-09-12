"""
Smart CCTV Monitoring System — Streamlit dashboard.

Three input modes (Webcam / Video Upload / RTSP) all feed the SAME
detector -> tracker -> rule-engine pipeline (see video_processor.py).
"""

import os
import sys
import tempfile
import threading
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import PersonDetector
from app.tracker import CentroidIoUTracker
from app.rules import RuleEngine
from app.video_processor import VideoProcessor, process_video_file
from app.video_sources import VideoSourceError

st.set_page_config(page_title="Smart CCTV Monitor", layout="wide")

MAX_UPLOAD_MB = 100
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@st.cache_resource
def load_detector():
    return PersonDetector(conf_threshold=0.4)


def sidebar_rule_config():
    st.sidebar.header("Rule Configuration")
    max_occupancy = st.sidebar.number_input(
        "Max occupancy (0 = disabled)", min_value=0, value=0, step=1
    )
    enable_zone = st.sidebar.checkbox("Enable restricted zone (fixed demo rectangle)")
    zone = []
    if enable_zone:
        st.sidebar.caption("Demo zone: top-left rectangle of the frame.")
        zone = [(0, 0), (200, 0), (200, 200), (0, 200)]
    return RuleEngine(restricted_zone=zone, max_occupancy=max_occupancy)


class LiveProcessor(VideoProcessorBase):
    """
    IMPORTANT: recv() runs on a background WebRTC worker thread, NOT
    Streamlit's main script thread. Calling st.* functions (st.markdown,
    st.toast, etc.) from inside recv() is unsupported — it silently does
    nothing (or warns), so any UI update attempted there never appears.
    This is why an earlier version of this file showed the raw, unprocessed
    webcam feed with no boxes/HUD: the annotated frame was computed
    correctly, but the crash/no-op happened on the stats-update call that
    followed it in the same method, before the annotated frame was returned
    in one code path.

    Fix: recv() only computes results and stores them behind a lock. The
    main thread polls that shared state in a loop and does the st.* calls.
    """

    def __init__(self, processor):
        self.processor = processor
        self._lock = threading.Lock()
        self.latest_counts = {"current_count": 0, "unique_count": 0}
        self._alert_queue = []  # drained by the main-thread poll loop, never dropped

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            result = self.processor.process_frame(img)
        except Exception:
            cv2.putText(img, "Frame error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        with self._lock:
            self.latest_counts = {
                "current_count": result["current_count"],
                "unique_count": result["unique_count"],
            }
            # Queued (not overwritten) so a poll interval slower than the
            # frame rate can't silently drop an alert that only appears
            # on a single deduplicated frame.
            self._alert_queue.extend(result["new_alerts"])

        return av.VideoFrame.from_ndarray(result["annotated_frame"], format="bgr24")

    def get_latest_counts(self):
        with self._lock:
            return dict(self.latest_counts)

    def drain_alerts(self):
        """Pop and return all alerts queued since the last drain."""
        with self._lock:
            alerts, self._alert_queue = self._alert_queue, []
            return alerts


def render_webcam_mode(rules):
    st.subheader("Webcam Monitoring")
    st.caption("Runs entirely in your browser session via WebRTC.")

    detector = load_detector()
    tracker = CentroidIoUTracker()
    processor = VideoProcessor(detector, tracker, rules)

    ctx = webrtc_streamer(
        key="webcam-monitor",
        video_processor_factory=lambda: LiveProcessor(processor),
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )

    stats_placeholder = st.empty()
    alerts_placeholder = st.empty()
    history_placeholder = st.expander("Event history")

    # Polling loop runs on the MAIN thread (safe for st.* calls) while the
    # stream is active, reading whatever the background recv() thread last stored.
    while ctx.state.playing:
        if ctx.video_processor:
            counts = ctx.video_processor.get_latest_counts()
            stats_placeholder.markdown(
                f"**Current people:** {counts['current_count']} &nbsp;&nbsp; "
                f"**Unique so far:** {counts['unique_count']}"
            )
            new_alerts = ctx.video_processor.drain_alerts()
            if new_alerts:
                alerts_placeholder.warning(" | ".join(new_alerts))
                processor.event_history.extend(new_alerts)
        time.sleep(0.3)

    with history_placeholder:
        for e in processor.event_history:
            st.write(e)


def render_upload_mode(rules):
    st.subheader("Video Upload")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is None:
        return

    ext = os.path.splitext(uploaded.name)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        st.error(f"Unsupported file type: {ext}")
        return

    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"File too large ({size_mb:.1f} MB). Limit is {MAX_UPLOAD_MB} MB.")
        return

    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp_in.write(uploaded.read())
    tmp_in.close()
    tmp_out_path = tmp_in.name + "_annotated.mp4"

    detector = load_detector()
    tracker = CentroidIoUTracker()

    progress_bar = st.progress(0.0)
    status = st.empty()

    def on_progress(frame_idx, total):
        if total > 0:
            progress_bar.progress(min(frame_idx / total, 1.0))
        status.text(f"Processing frame {frame_idx}/{total or '?'}")

    try:
        with st.spinner("Analyzing video..."):
            stats = process_video_file(
                tmp_in.name, tmp_out_path, detector, tracker, rules,
                progress_callback=on_progress,
            )
    except VideoSourceError as e:
        st.error(f"Could not process video: {e}")
        os.unlink(tmp_in.name)
        return
    except Exception as e:
        st.error("An unexpected error occurred while processing the video.")
        os.unlink(tmp_in.name)
        return

    status.text("Done.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Frames processed", stats["frames_processed"])
    col2.metric("Peak concurrent people", stats["max_current_count"])
    col3.metric("Unique people (approx.)", stats["unique_count"])

    st.video(tmp_out_path)
    with open(tmp_out_path, "rb") as f:
        st.download_button("Download annotated video", f, file_name="annotated_output.mp4")

    if stats["event_history"]:
        with st.expander("Event history"):
            for e in stats["event_history"]:
                st.write(e)

    os.unlink(tmp_in.name)
    os.unlink(tmp_out_path)


def render_rtsp_mode(rules):
    st.subheader("CCTV / RTSP Stream")
    st.info(
        "This app runs on a public cloud server, not on your local network. "
        "It can only reach an RTSP camera that is itself reachable from the "
        "public internet (e.g. via port-forwarding, a VPN gateway, or a "
        "cloud-relay camera service). It cannot automatically see a camera "
        "that only exists on your home/office LAN."
    )
    rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://user:pass@host:554/stream")
    start = st.button("Connect")

    if not start or not rtsp_url:
        return

    detector = load_detector()
    tracker = CentroidIoUTracker()
    processor = VideoProcessor(detector, tracker, rules)

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    stop = st.button("Stop")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        st.error("Could not open RTSP stream. Check the URL and network reachability.")
        return

    try:
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.warning("Stream ended or connection lost.")
                break
            result = processor.process_frame(frame)
            frame_placeholder.image(
                cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
            )
            stats_placeholder.markdown(
                f"**Current people:** {result['current_count']} &nbsp;&nbsp; "
                f"**Unique so far:** {result['unique_count']}"
            )
    finally:
        cap.release()


def main():
    st.title("Smart CCTV Monitoring System")
    st.caption(
        "AI monitoring layer over video feeds: detects people, tracks them, "
        "and raises alerts for configurable zone/occupancy rules. "
        "It does not judge intent — only measurable events."
    )

    rules = sidebar_rule_config()
    mode = st.sidebar.radio("Input source", ["Webcam", "Video Upload", "RTSP"])

    if mode == "Webcam":
        render_webcam_mode(rules)
    elif mode == "Video Upload":
        render_upload_mode(rules)
    else:
        render_rtsp_mode(rules)


if __name__ == "__main__":
    main()
