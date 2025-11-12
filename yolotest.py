import cv2
from ultralytics import YOLO
import os
import torch

VIDEO_PATH = "traffic.mp4"
TEMP_OUTPUT = "temp_output.mp4"
FINAL_OUTPUT = "sim1test11-11.mp4"

# use GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device.upper()}")

# Load YOLO model onto device (ideally cuda)
model = YOLO("yolo11l.pt")
model.to(device)

# Only classes that matter
TARGET_CLASSES = ["car", "bus", "truck", "traffic light"]

# Assign colors per class (BGR) for diffrentiaton/display purposes
CLASS_COLORS = {
    "person": (255, 0, 0),        # Blue
    "car": (0, 0, 255),           # Red
    "bus": (0, 255, 0),           # Green
    "truck": (0, 255, 0),         # Green (same as bus)
    "traffic light": (0, 255, 255) # Yellow
}

# Read video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Temporary output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(TEMP_OUTPUT, fourcc, fps, (width, height))

print("Press 'q' to stop.")

# Label settings 
BOX_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
LABEL_BG_COLOR = (255, 255, 255)  # white background for label text
LABEL_TEXT_COLOR = (0, 0, 0)      # makes text black

# Process video 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label in TARGET_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Combine bus and truck for display (since firetruck looks like both)
            display_label = "bus/truck" if label in ["bus", "truck"] else label

            # Use color based on the original label
            color = CLASS_COLORS.get(label, (255, 255, 255))  # default white

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            # Prepare label background and text
            text = f"{display_label} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width + 4, y1), LABEL_BG_COLOR, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 4), FONT, FONT_SCALE, LABEL_TEXT_COLOR, FONT_THICKNESS)

    # Show frame
    cv2.imshow("YOLO Detection", frame)
    out.write(frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup 
cap.release()
out.release()
cv2.destroyAllWindows()

# Save if wanted 
choice = input("Save the processed video? (y/n): ").strip().lower()
# Save processed video safely
if choice == "y":
    if os.path.exists(FINAL_OUTPUT):
        os.remove(FINAL_OUTPUT)  # remove existing file first
    os.rename(TEMP_OUTPUT, FINAL_OUTPUT)
    print(f"Saved as {FINAL_OUTPUT}")
else:
    os.remove(TEMP_OUTPUT)
    print("Deleted temporary video.")
