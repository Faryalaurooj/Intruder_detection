import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
cap = cv2.VideoCapture("input/11.mp4") #5,11

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame to let the user select the ROI
ret, frame = cap.read()
if not ret:
    print("Error: Failed to read the first frame.")
    exit()

# Ask the user to draw a box (ROI) around the area of interest (intruder detection zone)
roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Generate a unique filename using timestamp
output_video_path = f"output_intruder/annotated_video_{int(time.time())}.mp4"

# Initialize the VideoWriter to save the output video

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20, (640, 480))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Draw the ROI box on each frame to make it stay throughout the video
    x, y, w, h = roi  # Extract the coordinates of the ROI
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Run YOLOv8 detection
    results = model(frame)

    # Loop through the detections and check if they are inside the ROI
    for track in results[0].boxes:
        if track.cls == 0:  # Class 0 is for 'person'
            # Get bounding box coordinates
            xyxy = track.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Check if the person is inside the selected ROI
            if x1 >= x and y1 >= y and x2 <= x + w and y2 <= y + h:
                cv2.putText(frame, "Intruder", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw box around the person

    # Display the result and save the output
    out.write(frame)
    cv2.imshow("Intruder Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_video_path}")

