# Smart CCTV Monitoring System

An AI monitoring layer over video feeds: detects people, tracks them across
frames, and raises alerts for configurable rules (restricted-zone intrusion,
occupancy limits). Built as an evolution of a 2nd-year college project that
used a classical HOG+SVM detector on a single uploaded video with no
tracking, rules, or UI.

## Real-World Problem

Manually watching CCTV feeds continuously is tiring and error-prone; important
events get missed. This system acts as an always-on AI layer that watches the
feed and surfaces **measurable events** to a human operator — it does not
claim to judge whether a person is "suspicious," only whether they entered a
defined zone or occupancy was exceeded.

## Target User

A small facility, office, or store operator who wants a lightweight
person-counting/zone-alert layer without an expensive commercial CCTV
analytics platform.

## Features

- Person detection (YOLOv8n, COCO `person` class only)
- Bounding boxes + stable per-person tracking IDs
- Live people count (current) and unique-people count (cumulative)
- Restricted-zone intrusion alerts (polygon zone)
- Occupancy threshold alerts
- Live alert display + event history
- Three input modes sharing one pipeline: **Webcam**, **Video Upload**, **RTSP**
- Annotated output video + download for uploaded videos
- Graceful error handling (bad files, unreachable streams, model errors)

Not included yet (explicitly deferred): loitering detection, email/push
notifications, advanced analytics.

## Architecture

```
             VIDEO SOURCES
      ┌──────────┼──────────┐
      ↓          ↓          ↓
   Webcam     Video File    RTSP
      └──────────┼──────────┘
                 ↓
           Frame Processing
                 ↓
           Object Detection (YOLOv8n)
                 ↓
            Person Filtering (class=person, conf>=0.4)
                 ↓
               Tracking (IoU-based ID assignment)
                 ↓
             Rule Engine (zone / occupancy)
                 ↓
          Alerts + Statistics
                 ↓
              Dashboard (Streamlit)
```

All three input modes are just different frame *sources*
(`app/video_sources.py`); they feed the identical
detect → track → rule pipeline (`app/video_processor.py`), so there is
exactly one copy of the AI logic to test and maintain.

## Technologies

- **Detection:** YOLOv8n via `ultralytics` — nano variant (~6MB weights),
  chosen for CPU-friendly inference speed while staying accurate enough for
  a person-detection MVP. Pretrained on COCO, so class 0 (`person`) works
  out of the box with no custom training.
- **Tracking:** custom lightweight IoU-based tracker (`app/tracker.py`) —
  matches each new detection to the closest-overlapping existing track;
  survives brief occlusion (configurable `max_missed` frames); assigns a new
  ID otherwise. No external tracking library, so the logic is fully
  inspectable and explainable.
- **Rule engine:** `app/rules.py` — point-in-polygon zone check +
  occupancy-count comparison, deliberately decoupled from detection/tracking.
- **Dashboard:** Streamlit + `streamlit-webrtc` (for in-browser webcam
  access without a separate media server).
- **Video I/O:** OpenCV (`cv2.VideoCapture` / `cv2.VideoWriter`) — same API
  works for webcam index, file path, and RTSP URL.

## Detection Pipeline

Frame → YOLOv8n inference (person class only) → confidence filter (≥0.4) →
bounding boxes → tracker update → rule evaluation → annotated frame.

**Concept check — detection vs. tracking:**
- Detection asks: *"What objects exist in this frame?"* (independent per frame)
- Tracking asks: *"Is this the same person I saw in previous frames?"*
- **current_count** = people visible in the current frame.
- **unique_count** = number of distinct track IDs ever seen (total individuals
  across the whole video/session) — these are different numbers and the code
  keeps them separate on purpose.

## Tracking

Simple, explainable IoU-matching tracker (see `app/tracker.py`). Verified by
automated tests (`tests/test_tracker.py`) for: ID persistence across small
movement, ID survival across brief occlusion, distinct IDs for distinct
people, and track removal after prolonged absence.

## Rule Engine

- **Restricted Zone:** configurable polygon; a tracked person's box-center
  entering the polygon triggers `RESTRICTED AREA INTRUSION`.
- **Occupancy:** configurable max; `current_count > max` triggers
  `OCCUPANCY ALERT`.
- **Loitering:** not implemented (deferred per project plan — would need
  per-track dwell-time timers layered on the existing zone check).

## Installation

```bash
git clone <your-repo-url>
cd human-video-monitor
pip install -r requirements.txt
```

YOLOv8n weights (`yolov8n.pt`, ~6MB) are downloaded automatically by
`ultralytics` on first run.

## Local Usage

```bash
streamlit run app/app.py
```

Then open the printed local URL in a browser.

## Input Sources

- **Webcam:** uses your browser's camera via WebRTC — works locally and
  when deployed publicly (the browser streams to the app, not the other way
  around).
- **Video Upload:** upload `.mp4/.avi/.mov/.mkv` (≤100MB by default); the app
  processes the whole file and returns an annotated video + stats.
- **RTSP:** enter a stream URL. **Important networking note:** a publicly
  deployed instance of this app runs on cloud infrastructure, not on your
  home/office network. It can only reach an RTSP camera that is itself
  reachable from the public internet (port-forwarded, behind a VPN gateway,
  or via a cloud-relay camera service). It cannot automatically discover or
  reach a camera that only exists on a private LAN.

## Example Results

Ran against `ultralytics`' bundled sample image (`zidane.jpg`, two real
people): detector correctly found both, at confidence 0.84 and 0.82.
Full pipeline test (detect → track → rules → annotate) on a 15-frame test
clip: 2 people tracked with stable IDs throughout, occupancy alert fired
once (deduplicated) when a limit of 1 was set.

## Performance Measurements

Measured in this development sandbox (single CPU core, no GPU), at
1280×720 resolution, YOLOv8n, batch size 1:

- **~50 ms/frame inference → ~20 FPS** (detector only, no I/O overhead)

This is a sandbox measurement, not a guarantee for any specific deployment
target — actual FPS on a given free-tier host depends on that host's CPU
allocation and will typically be lower under concurrent load. Re-measure
after deploying.

## Deployment

**Target platform: Hugging Face Spaces** (free CPU tier) — chosen for native
Streamlit support, sufficient free RAM/CPU for YOLOv8n, and a public URL
with no separate hosting setup.

Steps:
1. Create a new Space at huggingface.co/new-space, SDK = **Streamlit**.
2. Push this repository's contents to the Space's git remote (Spaces are
   git repos).
3. Ensure `requirements.txt` is at the repo root (Spaces installs it
   automatically).
4. Set the Space's app file to `app/app.py` (Space settings, or `app_file`
   in the Space's `README.md` YAML header, e.g.:
   ```yaml
   ---
   sdk: streamlit
   app_file: app/app.py
   ---
   ```
   ).
5. Wait for the build; the Space will expose a public URL once running.
6. Test all three modes on the deployed URL — webcam and RTSP behavior can
   differ from local testing (browser permissions, outbound network rules).

## Limitations

- Single-class (person) detection only; no re-identification across camera
  restarts or after a track is dropped.
- Tracker uses IoU matching with a centroid-distance fallback (to survive
  low effective frame rate on CPU — see `app/tracker.py` docstring for why).
  It's still simple and can be fooled by two people crossing paths closely
  or swapping positions quickly; not benchmarked against MOT metrics.
- RTSP mode requires a publicly reachable camera, as noted above.
- No loitering detection, notifications, or authentication/access control —
  not suitable as-is for a real security deployment without those additions.
- Free-tier CPU hosting means limited FPS and possible cold-start delay.

## Future Improvements

- Loitering detection (dwell-time per zone)
- Stronger tracker (e.g. ByteTrack) if ID-switch rate proves too high
- Email/push notifications for alerts
- Multi-camera support
- Basic auth for the dashboard before any real-world use
