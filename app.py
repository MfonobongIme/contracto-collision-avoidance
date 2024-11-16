import streamlit as st
import cv2
import numpy as np
import torch
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Function to cache the model
@st.cache_resource
def load_model(model_path):
    return torch.hub.load(
        "yolov5-master/yolov5-master",
        "custom",
        model_path,
        source="local"
    )

# Set up the sidebar
st.sidebar.title("Settings")

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6)

# Video feed input
video_source = st.sidebar.text_input("Enter Video Feed URL or Path", 'test-video-trim.mp4')

# Load the YOLOv5 model only once (cached)
path = "yolov5-master/yolov5-master/yolov5n.pt"
model = load_model(path)
model.conf = confidence_threshold  # Set confidence threshold

# Open the video capture
cap = cv2.VideoCapture(video_source)

# Read a sample frame to set as the canvas background
ret, sample_frame = cap.read()
if not ret:
    st.error("Failed to read the video. Check your video source.")
    cap.release()
else:
    sample_frame_pil = Image.fromarray(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
    cap.release()
    cap = cv2.VideoCapture(video_source)

# Sidebar canvas for drawing ROI
st.sidebar.write("Draw Collision Area (ROI)")

# Initialize the canvas for freehand drawing mode
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 0, 0.3)",
    stroke_width=8,
    stroke_color="yellow",
    background_color="rgba(0, 0, 0, 0)",
    background_image=sample_frame_pil if ret else None,
    update_streamlit=True,
    height=sample_frame.shape[0] if ret else 480,
    width=sample_frame.shape[1] if ret else 640,
    drawing_mode="freedraw",  # Set to freehand drawing mode
    key="canvas",
)

# Initialize person_roi to avoid "name not defined" errors
person_roi = []

if st.sidebar.button("Submit ROI"):
    # Extract points from the canvas JSON data for the freehand drawing
    if canvas_result.json_data:
        if 'objects' in canvas_result.json_data:
            person_roi = []
            for obj in canvas_result.json_data["objects"]:
                # Ensure the object type is "path"
                if obj["type"] == "path":
                    # Loop through each point in the path
                    for point in obj["path"]:
                        # Ensure the point is in the format [<command>, x, y]
                        if len(point) == 3 and isinstance(point[1], (int, float)) and isinstance(point[2], (int, float)):
                            x, y = point[1], point[2]
                            person_roi.append((x, y))
            
            if person_roi:
                st.sidebar.write("Freehand ROI submitted:", person_roi)
            else:
                st.sidebar.write("No valid points extracted from the freehand ROI.")
        else:
            st.sidebar.write("No objects found in the canvas JSON data.")
    else:
        st.sidebar.write("No data captured from the canvas.")

# Read labels file
with open('coco2.txt', "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

# Stream video feed
stframe = st.empty()

# Add Start/Stop Video Button to control the loop
start_video = st.sidebar.button("Start Video")

if start_video:
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to read frame.")
            break

        trans = np.zeros_like(frame, np.uint8)

        # Set the model confidence threshold
        model.conf = confidence_threshold

        # Draw the ROI rectangle on the frame
        if person_roi:
            roi_array = np.array(person_roi, dtype=np.int32)
            # cv2.polylines(frame, [roi_array], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(trans, pts=[roi_array], color=(255, 255, 0))

        # Process the frame
        results = model(frame)
        person_detected_in_roi = False

        # Ensure person_roi is valid
        if person_roi:
            roi_array = np.array(person_roi, dtype=np.int32)
        else:
            roi_array = None

        for index, row in results.pandas().xyxy[0].iterrows():
            cl = row['class']
            xx1, xx2, yy1, yy2 = int(row['xmin']), int(row['xmax']), int(row['ymin']), int(row['ymax'])
            c = class_list[int(cl)] if int(cl) < len(class_list) else "unknown"

            if c == 'person' and roi_array is not None:
                # Calculate midpoint for the detected bounding box
                person_midpoint = ((xx1 + xx2) // 2, (yy1 + yy2) // 2)
                inside_person_roi = cv2.pointPolygonTest(roi_array, (xx2, yy2), False)

                if inside_person_roi > 0:
                    person_detected_in_roi = True
                    cv2.rectangle(trans, (xx1, yy1), (xx2, yy2), (0, 0, 255), -1)
                    cv2.rectangle(frame, (xx1, yy1), (xx2, yy2), (255, 255, 255), 2)
                    cv2.putText(frame, 'Person detected', (10, 30), 0, 1, (0, 0, 255), 2)
                    cv2.putText(frame, 'Suspend lift operations!!!!', (10, 70), 0, 1, (0, 0, 255), 2)

        # Combine the transparent frame with the main frame
        output_frame = frame.copy()
        alpha = 0.5
        mask = trans.astype(bool)
        output_frame[mask] = cv2.addWeighted(frame, alpha, trans, 1 - alpha, 0)[mask]

        # Display the frame
        stframe.image(output_frame, channels="BGR")

    # Release resources
    cap.release()

