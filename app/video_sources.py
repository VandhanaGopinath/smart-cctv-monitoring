"""
Video source abstraction.

Webcam, uploaded video file, and RTSP stream all become the same thing
to the rest of the pipeline: a generator that yields frames. This means
detector/tracker/rules code never needs to know where frames came from.

IMPORTANT (networking reality for RTSP):
A publicly hosted web app (e.g. on Hugging Face Spaces) runs on Anthropic/HF's
cloud infrastructure, NOT on your local network. It can only open an RTSP URL
if that camera is reachable from the public internet (e.g. via port forwarding,
a VPN, or a cloud-relay camera). It CANNOT automatically reach a camera that
only exists on your home/office LAN. This must be documented for users.
"""

import cv2


class VideoSourceError(Exception):
    """Raised when a video source cannot be opened or read."""
    pass


def frames_from_capture(cap, source_description="video source"):
    """
    Shared frame-generator logic for any cv2.VideoCapture-based source
    (webcam index, file path, or RTSP URL all use cv2.VideoCapture).
    """
    if not cap.isOpened():
        raise VideoSourceError(f"Could not open {source_description}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def webcam_source(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    return frames_from_capture(cap, f"webcam (index {camera_index})")


def file_source(video_path: str):
    cap = cv2.VideoCapture(video_path)
    return frames_from_capture(cap, f"video file '{video_path}'")


def rtsp_source(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url)
    return frames_from_capture(cap, f"RTSP stream '{rtsp_url}'")
