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

# Investigator prompt
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

# Load API key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
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

# Save uploaded file
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
tmp.write(video_file.read())
video_path = tmp.name

# Utility functions
def iou(a,b):
    xA,yA = max(a[0],b[0]), max(a[1],b[1])
    xB,yB = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,xB-xA)*max(0,yB-yA)
    areaA=(a[2]-a[0])*(a[3]-a[1])
    areaB=(b[2]-b[0])*(b[3]-b[1])
    return inter/float(areaA+areaB-inter+1e-6)

def call_ai(blocks):
    r=requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json"},
        json={"model":"gpt-4o","messages":[{"role":"user","content":blocks}],
              "response_format":{"type":"json_object"},"max_tokens":500}
    )
    if r.status_code!=200:
        st.warning(f"API error {r.status_code}")
        return {}
    c=r.json()['choices'][0]['message']['content']
    if isinstance(c,str):
        try: c=json.loads(c)
        except: return {}
    return c if isinstance(c,dict) else {}

# Analyze under spinner
with st.spinner("Analyzing..."):
    cap=cv2.VideoCapture(video_path)
    fps=cap.get(cv2.CAP_PROP_FPS)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Stage 1: motion & collision prescreen
    prev=None; scores=[]; colls=[]
    cap=cv2.VideoCapture(video_path); idx=0
    while True:
        ret,frame=cap.read()
        if not ret: break
        if idx%MOTION_INTERVAL==0:
            g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            g=cv2.GaussianBlur(g,(21,21),0)
            if prev is not None:
                d=cv2.absdiff(prev,g)
                s=np.sum(d>25)/(frame.shape[0]*frame.shape[1])*100; t=idx/fps
                scores.append((t,s))
                res=yolo.predict(frame,imgsz=640,conf=0.3,max_det=10)
                boxes=[b[:4] for r in res for b in r.boxes.data.tolist() if int(b[5]) in VEHICLE_CLASSES]
                hit=any(iou(boxes[a],boxes[b])>IOU_COLLISION_THRESH for a in range(len(boxes)) for b in range(a+1,len(boxes)))
                colls.append((t,hit))
            prev=g
        idx+=1
    cap.release()

    # Stage 2: select events
    motion_times=[t for t,s in scores if s>MOTION_THRESHOLD]
    coll_times=[t for t,h in colls if h]
    events=[t for t in motion_times if any(abs(t-c)<DETAILED_SECONDS for c in coll_times)]
    # minimal gap
    filt=[]
    for t in sorted(events):
        if not filt or t-filt[-1]>MIN_EVENT_GAP: filt.append(t)
    events=filt

    # Stage 3: AI verification & frame capture
    confirmed=[]
    for ct in events:
        # AI blocks
        blocks=[
            {"type":"text","text":"Please respond only with JSON."},
            {"type":"text","text":INVESTIGATOR_PROMPT}
        ]
        res=call_ai(blocks)
        if res.get('is_crash'):
            # grab frame at impact
            impact_idx=int(ct*fps)
            cap2=cv2.VideoCapture(video_path)
            cap2.set(cv2.CAP_PROP_POS_FRAMES,impact_idx)
            ok,frm=cap2.read()
            cap2.release()
            if ok:
                confirmed.append((ct,frm))
            else:
                confirmed.append((ct,None))
        time.sleep(API_CALL_DELAY_SECONDS)

# Final display
st.header("📊 Crash Detection Results")
if confirmed:
    for t,frm in confirmed:
        st.markdown(f"**Crash at {t:.2f}s**")
        if frm is not None:
            st.image(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB), use_column_width=True)
else:
    st.info("No crashes confirmed in the video.")
