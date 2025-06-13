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

# Load YOLOv8 model (cached)
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo = load_yolo()

st.title("🚗 Car Crash Detection App")
video_file = st.file_uploader("Upload a video file", type=["mp4","mov","avi"])

if not video_file:
    st.info("Awaiting video upload...")
    st.stop()

# Save upload to temp
suffix = os.path.splitext(video_file.name)[1]
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
tmp.write(video_file.read())
video_path = tmp.name

# Utility: IoU
def iou(a,b):
    xA,yA = max(a[0],b[0]), max(a[1],b[1])
    xB,yB = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,xB-xA)*max(0,yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter/float(areaA+areaB-inter+1e-6)

# Call AI
def call_ai(blocks):
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json"},
        json={"model":"gpt-4o","messages":[{"role":"user","content":blocks}],
              "response_format":{"type":"json_object"},"max_tokens":500}
    )
    if resp.status_code!=200:
        st.warning(f"API error {resp.status_code}: {resp.text}")
        return {}
    c = resp.json()['choices'][0]['message']['content']
    if isinstance(c,str):
        try: c=json.loads(c)
        except: return {}
    return c if isinstance(c,dict) else {}

# Load video info
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

st.info(f"Loaded: {fps:.1f} FPS, {frames} frames")
bar = st.progress(0)

# Stage 1: prescreen motion & collision
prev=None
scores=[]
colls=[]
cap=cv2.VideoCapture(video_path)
i=0
while True:
    ret,frame=cap.read()
    if not ret: break
    if i%MOTION_INTERVAL==0:
        g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        g=cv2.GaussianBlur(g,(21,21),0)
        if prev is not None:
            d=cv2.absdiff(prev,g)
            s=np.sum(d>25)/(frame.shape[0]*frame.shape[1])*100
            t=i/fps; scores.append((t,s))
            res=yolo.predict(frame,imgsz=640,conf=0.3,max_det=10)
            boxes=[b[:4] for r in res for b in r.boxes.data.tolist() if int(b[5]) in VEHICLE_CLASSES]
            hit=any(iou(boxes[a],boxes[b])>IOU_COLLISION_THRESH for a in range(len(boxes)) for b in range(a+1,len(boxes)))
            colls.append((t,hit))
        prev=g
    i+=1; bar.progress(min(i/frames,1.0))
cap.release()

# Stage 2: pick events
peaks=[scores[j][0] for j in range(1,len(scores)-1)
       if scores[j][1]>scores[j-1][1] and scores[j][1]>scores[j+1][1] and scores[j][1]>MOTION_THRESHOLD]
cts=[t for t,h in colls if h]
evt=sorted(set(peaks+cts))
fl=[]
for t in evt:
    if not fl or t-fl[-1]>MIN_EVENT_GAP: fl.append(t)
events=fl
if not events:
    st.success("No events detected.")
    st.stop()
else:
    st.success(f"Found {len(events)} events")

conf=[]
for idx,ct in enumerate(events):
    st.write(f"#### Event {idx+1} @ {ct:.2f}s")
    start=max(0,ct-DETAILED_SECONDS/2)
    sf=int(start*fps)
    cap=cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
    acc=None; pg=None; smalls=[]; raws=[]
    for j in range(int(DETAILED_SECONDS*fps)):
        ok,fm=cap.read()
        if not ok: break
        if j%max(1,int(fps/DETAILED_FPS))==0:
            h,w=fm.shape[:2]; nh=int(FRAME_RESIZE_WIDTH*h/w)
            sm=cv2.resize(fm,(FRAME_RESIZE_WIDTH,nh))
            smalls.append(sm); raws.append(fm)
            gg=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY)
            gg=cv2.GaussianBlur(gg,(21,21),0)
            if pg is not None:
                dv=cv2.absdiff(pg,gg)
                acc=dv if acc is None else acc+dv
            pg=gg
    cap.release()
    if acc is None: continue
    # heatmaps & ROI & collision mask & frames to b64 omitted for brevity, same logic
    # build blocks
    blocks=[
        {"type":"text","text":"Please respond in valid json format."},
        {"type":"text","text":INVESTIGATOR_PROMPT},
        # ... more multimodal blocks ...
    ]
    res=call_ai(blocks)
    if 'is_crash' not in res:
        st.warning("Empty response, defaulting crash")
        res={"is_crash":True,"impact_frame_index":None,"reasoning":"fallback"}
    st.write(f"**Crash?** {res['is_crash']} – {res['reasoning']}")
    if res['is_crash']:
        conf.append((ct,raws[res.get('impact_frame_index') or 0]))
    time.sleep(API_CALL_DELAY_SECONDS)

# Final
st.header("Results")
if conf:
    for t,frm in conf:
        st.write(f"🚨 Crash @ {t:.2f}s")
        st.image(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))
else:
    st.success("No crashes confirmed.")
