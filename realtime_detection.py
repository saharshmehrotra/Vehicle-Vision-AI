import cv2
import torch

# Load the YOLOv5 model (change to your own model if needed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Use GPU if available
if torch.cuda.is_available():
    model.cuda()
    print("Running on GPU")
else:
    print("Running on CPU")

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Convert to RGB and run detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)

    # Draw results
    annotated_frame = results.render()[0]

    # Show frame
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

