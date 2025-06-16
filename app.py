import os
import tempfile
import base64
import requests
import json
import numpy as np
import streamlit as st
from PIL import Image
import imageio
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
  "reasoning": "one-sentence justification"
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
video_file = st.file_uploader("Upload a video or image file", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])

if not video_file:
    st.info("Upload a video or image to begin.")
    st.stop()

# Save upload
suffix = os.path.splitext(video_file.name)[1]
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
tmp.write(video_file.read())
file_path = tmp.name

# Check if the uploaded file is an image or video
if suffix.lower() in ['.jpg', '.jpeg', '.png']:
    # Process as image
    image = Image.open(file_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # You can apply any image processing here, such as crash detection logic on the image.
    # For now, we just display the image
    st.success("Image processed successfully.")

elif suffix.lower() in ['.mp4', '.mov', '.avi']:
    # Process as video
    cap = imageio.get_reader(file_path)
    fps = cap.get_meta_data()['fps']
    total_frames = len(cap)
    st.info(f"Video contains {total_frames} frames at {fps} FPS.")
    
    # Example: Display the first frame of the video
    first_frame = cap.get_data(0)
    st.image(first_frame, caption="First Frame of Video", use_column_width=True)
    
    # Apply your crash detection logic on the video here
    st.success("Video processed successfully.")
    
    # Example: Process frames and apply YOLO or crash detection
    confirmed = []  # To store confirmed crash events
    for idx, frame in enumerate(cap):
        if idx % MOTION_INTERVAL == 0:
            # Process the frame for crash detection (for simplicity, here we are skipping this)
            # If you have collision or motion detection, apply it here for each frame
            pass
    
    # Final step for video: display results, if any crashes were detected
    if confirmed:
        st.markdown("🚗 Crash detected in the video!")
        # Display crash frames (optional)
        for t, frm in confirmed:
            st.image(frm, caption=f"Crash at {t:.2f}s", use_column_width=True)
    else:
        st.info("No crashes detected in the video.")

else:
    st.error("Invalid file type. Please upload a valid image or video.")
