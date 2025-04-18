import cv2
import numpy as np

# Input video paths
left_video_path = "/export/scratch/ra23mar/test/EDGS/docs/static/videos/lego_3d.mp4"   # e.g., 3DGS
right_video_path = "/export/scratch/ra23mar/test/EDGS/docs/static/videos/lego.mp4" # e.g., EDGS

# Load videos
cap1 = cv2.VideoCapture(left_video_path)
cap2 = cv2.VideoCapture(right_video_path)

# Get properties (assuming both videos have the same height and fps)
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap1.get(cv2.CAP_PROP_FPS)

# Output video writer (double the width)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("combined_output.mp4", fourcc, fps, (width * 2, height))

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
thickness = 4
color = (255, 255, 255)  # white

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Resize to same size if needed
    frame2 = cv2.resize(frame2, (width, height))

    # Add "3DGS" top-left of left video
    cv2.putText(frame1, "3DGS", (30, 60), font, font_scale, color, thickness, cv2.LINE_AA)

    # Add "EDGS" top-right of right video
    text_size, _ = cv2.getTextSize("EDGS", font, font_scale, thickness)
    text_x = width - text_size[0] - 30
    cv2.putText(frame2, "EDGS", (text_x, 60), font, font_scale, color, thickness, cv2.LINE_AA)

    # Combine horizontally
    combined = np.hstack((frame1, frame2))

    out.write(combined)

cap1.release()
cap2.release()
out.release()
print("✅ Combined video saved as combined_output.mp4")
