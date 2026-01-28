import cv2
from ultralytics import YOLO
import os
import torch
import time

start_time = time.time()

video_path = "benchmark3_60.mp4"
temp = "temp_output.mp4"
output_path = "sim1test_3.mp4"

# Make sure using GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device.upper()}")

# Load YOLO model
model = YOLO(r"runs\detect\train5\weights\best.pt")
model.to(device)

# Classes that matter
target_classes = ["ES_vehicle", "bike", "car", "pedestrian", "traffic_light", "truck_bus"]

# Colors for each class
class_colors = {
    "ES_vehicle": (255, 0, 0),       # Blue
    "bike": (0, 0, 255),             # Red
    "car": (0, 255, 0),              # Green
    "pedestrian": (255, 0, 255),     # Magenta
    "traffic_light": (0, 255, 0),    # Green
    "truck_bus": (0, 255, 255)       # Yellow
}

# Read video
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output at 30 FPS
output_fps = 30.0
frame_skip = 7

print(f"Input: {width}x{height} @ {input_fps:.1f}fps, {total_frames} frames")
print(f"Output: {width}x{height} @ {output_fps:.1f}fps")
print(f"Reading every {frame_skip} frames (skipping {frame_skip-1} frames)")

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp, fourcc, output_fps, (width, height))

print("Press 'q' to stop watching.")

# Label settings
box_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 2
label_bg_color = (255, 255, 255)
label_color = (0, 0, 0)

# FPS tracking
fps_list = []
processed_count = 0

# Display window setup
scale_percent = 50
display_width = int(width * scale_percent / 100)
display_height = int(height * scale_percent / 100)

def calculate_iou(box1, box2):
    # Calculate Intersection over Union between two boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    intersection = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    
    # Union area
    box1_area = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    box2_area = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def filter_nested_boxes(boxes_data, iou_threshold=0.3):
    # Remove nested/overlapping boxes, keeping the larger bounding box for each region.
    
    if len(boxes_data) <= 1:
        return boxes_data
    
    # Compute area for sorting
    for bd in boxes_data:
        x1, y1, x2, y2 = bd['xyxy']
        bd['area'] = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Sort by area (largest first) so larger boxes are considered before smaller nested ones
    boxes_sorted = sorted(boxes_data, key=lambda x: x['area'], reverse=True)
    
    filtered = []
    for box_data in boxes_sorted:
        box = box_data['xyxy']
        keep = True
        
        for kept_box_data in filtered:
            kept_box = kept_box_data['xyxy']
            iou = calculate_iou(box, kept_box)
            if iou > iou_threshold:
                # overlap with a larger kept box, skip this (smaller) one
                keep = False
                break
        
        if keep:
            filtered.append(box_data)

    for bd in filtered:
        if 'area' in bd:
            del bd['area']
    
    return filtered

# Process video... jump to frames instead of reading all
frame_position = 0
while frame_position < total_frames:
    # set frame position directly (skip reading unwanted frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
    
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_count += 1
    inference_start = time.time()
    
    # Run tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.3,          
        iou=0.3,           
        imgsz=416,
        half=True,
        device=device,
        verbose=False,
        max_det=100,
        tracker="bytetrack.yaml"
    )[0]
    
    # Calculate inference FPS
    inference_time = time.time() - inference_start
    current_fps = 1 / inference_time if inference_time > 0 else 0
    fps_list.append(current_fps)
    if len(fps_list) > 30:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    
    # Collect boxes data
    boxes_data = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label in target_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            boxes_data.append({
                'xyxy': (x1, y1, x2, y2),
                'conf': conf,
                'label': label,
                'cls_id': cls_id
            })
    
    # Filter out nested/duplicate boxes (keeps the larger box by area)
    boxes_data = filter_nested_boxes(boxes_data, iou_threshold=0.3)
    
    # Draw filtered bounding boxes
    for box_data in boxes_data:
        x1, y1, x2, y2 = box_data['xyxy']
        label = box_data['label']
        conf = box_data['conf']
        
        # Combine bus and truck for display
        display_label = "bus/truck" if label in ["bus", "truck"] else label
        color = class_colors.get(label, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        # Label
        text = f"{display_label} {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width + 4, y1), label_bg_color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), font, font_scale, label_color, font_thickness)

    # Add FPS counter
    cv2.putText(frame, f"Inference FPS: {avg_fps:.1f} | Frame: {processed_count}/{total_frames//frame_skip}", 
                (10, 30), font, 0.6, (0, 255, 0), 2)
    
    # Display
    resized_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    cv2.imshow("YOLO Detection", resized_frame)
    
    # Write output
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Jump to next frame
    frame_position += frame_skip

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Stats
end_time = time.time()
total_time = end_time - start_time
avg_fps_final = sum(fps_list) / len(fps_list) if fps_list else 0

print(f"\nProcessing complete.")
print(f"Frames processed: {processed_count}")
print(f"Average inference FPS: {avg_fps_final:.1f}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Processing speed: {processed_count / total_time:.1f} frames/sec")

# Save option
choice = input("\nSave the processed video? (y/n): ").strip().lower()
if choice == "y":
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp, output_path)
    print(f"Saved output as: {output_path}")
else:
    os.remove(temp)
    print("Deleted temporary video")
