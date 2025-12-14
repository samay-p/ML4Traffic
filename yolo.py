import cv2
from ultralytics import YOLO
import os
import torch

video_path = "traffic.mp4"
temp = "temp_output.mp4"
output_path = "sim1test11-11.mp4"

# make sure using GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device.upper()}")

# Load YOLO model onto gpu (if available)
model = YOLO("yolo11l.pt") # using yolo 11 large
model.to(device)

# classes that matter
target_classes = ["car", "bus", "truck", "traffic light", "person"]

# colors for each class... for easier diffrentiaton
class_colors = {
    "person": (255, 0, 0), # Blue
    "car": (0, 0, 255), # Red
    "bus": (0, 255, 0), # Green
    "truck": (0, 255, 0), # Green (same as bus)
    "traffic light": (0, 255, 255) # Yellow
}

# Read video
cap = cv2.VideoCapture(video_path)

# fixed errors w frame being too large
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Temporary output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp, fourcc, fps, (width, height))

print("Press 'q' to stop watching.")

# Label settings 
box_thickness = 3
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2
label_bg_color = (255, 255, 255) # White background for label text
label_color = (0, 0, 0) # Black text

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

        if label in target_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Combine bus and truck for display
            display_label = "bus/truck" if label in ["bus", "truck"] else label

            # Use color based on the original label
            color = class_colors.get(label, (255, 255, 255))  # default white

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            # Prepare label background and text
            text = f"{display_label} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width + 4, y1), label_bg_color, -1) # padding since previous label was distorted
            cv2.putText(frame, text, (x1 + 2, y1 - 4), font, font_scale, label_color, font_thickness)

    # Show frame
    cv2.imshow("YOLO Detection", frame)
    out.write(frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# end
cap.release()
out.release()
cv2.destroyAllWindows()

# allows for saving if wanted
choice = input("Save the processed video? (y/n): ").strip().lower()
# Save processed video safely
if choice == "y":
    if os.path.exists(output_path):
        os.remove(output_path)  # remove existing file first
    os.rename(temp, output_path)
    print(f"Saved output as{output_path}")
else:
    os.remove(temp)
    print("Deleted temporary vid")
