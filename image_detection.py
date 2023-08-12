import cv2
import torch
import numpy as np

# Load the YOLOv5 model from a checkpoint
path = 'C:/Users/HAMZA ABBAS/Desktop/insect_Detection_jupyter/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

# Load an image
image_path = 'images/test1.jpg'  # Replace with your image path
frame = cv2.imread(image_path)

# Resize the image to match the model's input size
frame = cv2.resize(frame, (640, 480))

# Perform object detection on the image
# results = model.model([frame])
results = model(frame)
# Wrap the frame in a list to create a batch with a single image

# Get the prediction results
pred = results.pred[0].cpu()

# Filter out detections with low confidence
conf_threshold = 0.1
filtered_detections = pred[pred[:, 4] > conf_threshold]

# Draw bounding boxes on the image
for x_min, y_min, x_max, y_max, conf, class_idx in filtered_detections:
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
    class_name = model.model.names[int(class_idx)]  # Access class names from model.model.names
    color = (0, 255, 0)  # Green color for the bounding box

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(frame, f"{class_name}: {conf:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the output image
cv2.imshow("Object Detection Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
