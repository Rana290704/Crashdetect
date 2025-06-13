import os
import tempfile
import base64
import requests
import json
import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Configuration constants
API_CALL_DELAY_SECONDS   = 20
FRAME_RESIZE_WIDTH       = 256
JPEG_QUALITY             = 50
MOTION_THRESHOLD         = 25
MIN_EVENT_GAP            = 0.5
MOTION_INTERVAL          = 2
DETAILED_SECONDS         = 6
DETAILED_FPS             = 12
MAX_SEND_FRAMES          = 8
IOU_COLLISION_THRESH     = 0.02
VEHICLE_CLASSES          = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Investigator prompt (must be defined before use)
INVESTIGATOR_PROMPT = """
You are an expert accident investigator. You will be given:

1. A global motion heatmap (6s clip) showing all movement intensity.
2. A cropped ROI heatmap of the single highest-motion area.
3. A collision mask image highlighting overlapping vehicle regions (if any).
4. Up to eight video frames from that period.

Your job is to decide if a real car crash occurred.
– Crashes show abrupt, concentrated spikes in motion AND visible vehicle deformation or overlap.
– Non-crashes (passing cars, camera pans) look smooth, uniform, or only minor overlaps.

**If you are uncertain based on the evidence, you MUST assume a crash occurred.**

**Respond ONLY with valid json** in this EXACT schema:
{
  "is_crash": bool,
  "impact_frame_index": int or null,
  "reasoning": "one-sentence justification referencing your evidence"
}
"""

# Load models & API key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

yolo = YOLO("yolov8n.pt")

st.title("🚗 Car Crash Detection App")
uploader = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])


def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)


def perform_ai_verification(blocks):
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": blocks}],
            "response_format": {"type": "json_object"},
            "max_tokens": 500
        }
    )
    if resp.status_code != 200:
        st.warning(f"API error {resp.status_code}: {resp.text}")
        return {}
    content = resp.json()['choices'][0]['message']['content']
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return {}
    return content if isinstance(content, dict) else {}


if uploader:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploader.name)[1])
    tfile.write(uploader.read())
    video_path = tfile.name

    # Read video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    st.info(f"Loaded video: {fps:.1f} FPS, {total_frames} frames")
    progress = st.progress(0)

    # 1) Prescreen: motion & collisions
    prev_gray = None
    raw_scores, raw_collisions = [], []
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % MOTION_INTERVAL == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21,21), 0)
            if prev_gray is not None:
                delta = cv2.absdiff(prev_gray, gray)
                score = np.sum(delta>25)/(frame.shape[0]*frame.shape[1])*100
                t = idx / fps
                raw_scores.append((t, score))
                results = yolo.predict(frame, imgsz=640, conf=0.3, max_det=10)
                boxes = [b[:4] for r in results for b in r.boxes.data.tolist() if int(b[5]) in VEHICLE_CLASSES]
                collision = any(iou(boxes[i], boxes[j]) > IOU_COLLISION_THRESH
                                for i in range(len(boxes)) for j in range(i+1, len(boxes)))
                raw_collisions.append((t, collision))
            prev_gray = gray
        idx += 1
        progress.progress(min(idx/total_frames, 1.0))
    cap.release()

    # 2) Determine events
    peaks = [raw_scores[i][0] for i in range(1, len(raw_scores)-1)
             if raw_scores[i][1] > raw_scores[i-1][1]
             and raw_scores[i][1] > raw_scores[i+1][1]
             and raw_scores[i][1] > MOTION_THRESHOLD]
    coll_times = [t for t, hit in raw_collisions if hit]
    event_times = sorted(set(peaks + coll_times))

    events = []
    for t in event_times:
        if not events or (t - events[-1]) > MIN_EVENT_GAP:
            events.append(t)

    if not events:
        st.success("No candidate events detected.")
        st.stop()
    else:
        st.success(f"Detected {len(events)} candidate events.")

    confirmed = []
    # 3) Detailed analysis
    for ei, center_t in enumerate(events):
        st.write(f"### Event {ei+1} at {center_t:.2f}s")
        start = max(0, center_t - DETAILED_SECONDS/2)
        sf = int(start * fps)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

        heat_acc, prev_g = None, None
        smalls, raw_frames = [], []
        for j in range(int(DETAILED_SECONDS * fps)):
            ret, frm = cap.read()
            if not ret:
                break
            if j % max(1, int(fps/DETAILED_FPS)) == 0:
                h, w = frm.shape[:2]
                nh = int(FRAME_RESIZE_WIDTH * h / w)
                small = cv2.resize(frm, (FRAME_RESIZE_WIDTH, nh), interpolation=cv2.INTER_AREA)
                smalls.append(small)
                raw_frames.append(frm)

                gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                gray_s = cv2.GaussianBlur(gray_s, (21,21), 0)
                if prev_g is not None:
                    dv = cv2.absdiff(prev_g, gray_s)
                    heat_acc = dv if heat_acc is None else heat_acc + dv
                prev_g = gray_s
        cap.release()

        if heat_acc is None:
            continue

        # Global heatmap
        hm_norm = cv2.normalize(heat_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        st.image(cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB), caption="Global Heatmap")
        _, buf = cv2.imencode(".jpg", hm_color)
        global_b64 = base64.b64encode(buf.tobytes()).decode()

        # ROI crop
        mask = hm_norm > (hm_norm.mean() + hm_norm.std())
        if mask.any():
            ys, xs = np.where(mask)
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            roi = hm_color[y1:y2, x1:x2]
            st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="ROI Heatmap")
            _, buf2 = cv2.imencode(".jpg", roi)
            roi_b64 = base64.b64encode(buf2.tobytes()).decode()
        else:
            roi_b64 = None

        # Collision mask
        mid = smalls[len(smalls)//2]
        res = yolo.predict(mid, imgsz=640, conf=0.3, max_det=10)
        bxs = [b[:4] for r in res for b in r.boxes.data.tolist() if int(b[5]) in VEHICLE_CLASSES]
        cm = np.zeros_like(mid[:,:,0])
        for i in range(len(bxs)):
            for j in range(i+1, len(bxs)):
                if iou(bxs[i], bxs[j]) > IOU_COLLISION_THRESH:
                    xA = int(max(bxs[i][0], bxs[j][0]))
                    yA = int(max(bxs[i][1], bxs[j][1]))
                    xB = int(min(bxs[i][2], bxs[j][2]))
                    yB = int(min(bxs[i][3], bxs[j][3]))
                    cm[yA:yB, xA:xB] = 255
        if cm.any():
            st.image(cm, caption="Collision Mask", clamp=True)
            _, buf3 = cv2.imencode(".jpg", cm)
            coll_b64 = base64.b64encode(buf3.tobytes()).decode()
        else:
            coll_b64 = None

        # Display sample frames
        st.write("Sample frames:")
        for s in smalls[:MAX_SEND_FRAMES]:
            st.image(cv2.cvtColor(s, cv2.COLOR_BGR2RGB))

        # Build blocks for AI
        blocks = [
            {"type":"text","text":"Please respond in valid json format."},
            {"type":"text","text":INVESTIGATOR_PROMPT},
            {"type":"text","text":"--- Global Heatmap ---"},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{global_b64}","detail":"low"}}
        ]
        if roi_b64:
            blocks += [
                {"type":"text","text":"--- ROI Heatmap ---"},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{roi_b64}","detail":"low"}}
            ]
        if coll_b64:
            blocks += [
                {"type":"text","text":"--- Collision Mask ---"},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{coll_b64}","detail":"low"}}
            ]
        blocks += [{"type":"text","text":"--- Video Frames ---"}]
        for s in smalls[:MAX_SEND_FRAMES]:
            _, bf = cv2.imencode(".jpg", s, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            b64 = base64.b64encode(bf.tobytes()).decode()
            blocks.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}","detail":"low"}})

        # AI verification
        result = perform_ai_verification(blocks)
        if 'is_crash' not in result:
            st.warning("Unexpected API response, defaulting to crash.")
            result = {"is_crash": True, "impact_frame_index": None, "reasoning": "fallback"}

        st.write(f"**AI verdict:** {result['is_crash']} – {result['reasoning']}")
        if result['is_crash']:
            imp = result.get('impact_frame_index') or 0
            confirmed.append((start + imp/DETAILED_FPS, raw_frames[imp]))

        if ei < len(events)-1:
            time.sleep(API_CALL_DELAY_SECONDS)

    # Final report
    st.header("Final Report")
    if confirmed:
        for t, frm in confirmed:
            st.write(f"🚨 Crash @ {t:.2f}s")
            st.image(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    else:
        st.success("No crashes confirmed.")
