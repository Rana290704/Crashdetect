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
IOU_COLLISION_THRESH     = 0.02
VEHICLE_CLASSES          = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Investigator prompt
INVESTIGATOR_PROMPT = """
You are an expert accident investigator. You will be given:

1. A motion heatmap image showing movement intensity over a 6-second clip.
2. A single video frame around the suspected impact.

Your job is to decide if a real car crash occurred.
– Look for abrupt, concentrated spikes in motion and vehicle deformation.
– Ignore smooth motion or minor overlaps.

**If uncertain, assume a crash occurred.**

Respond ONLY with valid JSON in this schema:
{
  "is_crash": bool,
  "impact_frame_index": int or null,
  "reasoning": "one-sentence justification",
  "accuracy": float  # New field for accuracy
}
"""

# Load API key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("Set the OPENAI_API_KEY environment variable.")
    st.stop()

# Cache YOLO model
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo = load_yolo()

st.title("🚗 Car Crash Detection App")
video_file = st.file_uploader("Upload a video file", type=["mp4","mov","avi"])
if not video_file:
    st.info("Upload a video to begin.")
    st.stop()

# Save upload
suffix = os.path.splitext(video_file.name)[1]
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
tmp.write(video_file.read())
video_path = tmp.name

# Utility: IoU
def iou(a,b):
    xA,yA = max(a[0],b[0]), max(a[1],b[1])
    xB,yB = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,xB-xA)*max(0,yB-yA)
    areaA=(a[2]-a[0])*(a[3]-a[1])
    areaB=(b[2]-b[0])*(b[3]-b[1])
    return inter/float(areaA+areaB-inter+1e-6)

# Call AI with JSON response
def call_ai(blocks):
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json"},
        json={"model":"gpt-4o","messages":[{"role":"user","content":blocks}]}  # Remove response_format
    )
    if resp.status_code != 200:
        return {}
    content = resp.json()['choices'][0]['message']['content']
    if isinstance(content,str):
        try: content = json.loads(content)
        except: return {}
    return content if isinstance(content,dict) else {}

# Analyze under spinner
with st.spinner("Analyzing video..."):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Stage 1: motion & collision prescreen
    scores, collisions = [], []
    prev_gray = None
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % MOTION_INTERVAL == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(21,21),0)
            if prev_gray is not None:
                delta = cv2.absdiff(prev_gray, gray)
                score = np.sum(delta>25)/(frame.shape[0]*frame.shape[1])*100
                t = idx / fps
                scores.append((t,score))
                # YOLO collision detect
                res = yolo.predict(frame,imgsz=640,conf=0.3,max_det=10)
                boxes = [b[:4] for r in res for b in r.boxes.data.tolist() if int(b[5]) in VEHICLE_CLASSES]
                hit = any(iou(boxes[i],boxes[j])>IOU_COLLISION_THRESH
                          for i in range(len(boxes)) for j in range(i+1,len(boxes)))
                collisions.append((t,hit))
            prev_gray = gray
        idx += 1
    cap.release()

    # Stage 2: pick events where motion & collision coincide
    high_motion = [t for t,s in scores if s> MOTION_THRESHOLD]
    coll_times = [t for t,h in collisions if h]
    events = [t for t in high_motion if any(abs(t-c)<DETAILED_SECONDS for c in coll_times)]
    # enforce gap
    filtered = []
    for t in sorted(events):
        if not filtered or t - filtered[-1] > MIN_EVENT_GAP:
            filtered.append(t)
    events = filtered

    # Stage 3: AI verify each event and collect frames
    confirmed = []
    for ct in events:
        # Generate motion heatmap
        cap = cv2.VideoCapture(video_path)
        start_frame = int(max(0,(ct - DETAILED_SECONDS/2)*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        acc = None
        prev_g = None
        for _ in range(int(DETAILED_SECONDS*fps)):
            ok, frm = cap.read()
            if not ok: break
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(21,21),0)
            if prev_g is not None:
                diff = cv2.absdiff(prev_g, gray)
                acc = diff if acc is None else acc + diff
            prev_g = gray
        cap.release()
        # create heatmap image
        hm_norm = cv2.normalize(acc,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        hm_img = cv2.applyColorMap(hm_norm,cv2.COLORMAP_JET)
        _, hm_buf = cv2.imencode('.jpg', hm_img)
        hm_b64 = base64.b64encode(hm_buf.tobytes()).decode()
        # grab impact frame
        frame_idx = int(ct*fps)
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, impact_frame = cap2.read()
        cap2.release()
        frame_b64 = ''
        if ok:
            _, fr_buf = cv2.imencode('.jpg', impact_frame, [int(cv2.IMWRITE_JPEG_QUALITY),JPEG_QUALITY])
            frame_b64 = base64.b64encode(fr_buf.tobytes()).decode()

        # build AI blocks
        blocks = []
        blocks.append({"type":"text","text":INVESTIGATOR_PROMPT})
        blocks.append({"type":"text","text":"--- Motion Heatmap ---"})
        blocks.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{hm_b64}","detail":"low"}})
        if frame_b64:
            blocks.append({"type":"text","text":"--- Impact Frame ---"})
            blocks.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{frame_b64}","detail":"low"}})

        res = call_ai(blocks)
        if res.get('is_crash'):
            confirmed.append((ct, impact_frame, res.get('accuracy')))
        time.sleep(API_CALL_DELAY_SECONDS)

# Final display
st.header("📊 Crash Detection Results")
if confirmed:
    for t, frm, accuracy in confirmed:
        st.markdown(f"**Crash at {t:.2f}s**")
        st.markdown(f"**Accuracy: {accuracy:.2f}%**")
        
        # Highlight accident area (rectangle)
        if frm is not None:
            # Example: Draw a rectangle on the frame (you need to define accident area coordinates)
            # Here I am assuming a placeholder rectangle; replace it with actual accident area
            accident_area = (100, 100, 400, 400)  # Example: (x1, y1, x2, y2)
            cv2.rectangle(frm, (accident_area[0], accident_area[1]), (accident_area[2], accident_area[3]), (0, 255, 0), 3)
            st.image(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    st.info("No crashes confirmed in the video.")
