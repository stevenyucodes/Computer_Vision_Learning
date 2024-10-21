
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Open webcam stream
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB and PIL format
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt")

    # Perform object detection
    with torch.no_grad():
        outputs = model(**inputs)

    # Get detected boxes and labels
    for i, score in enumerate(outputs.logits.softmax(-1)[0, :, :-1]):
        if score.max() > 0.5:  # Filter low-confidence predictions
            box = outputs.pred_boxes[i].detach().numpy()
            x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
            label = model.config.id2label[score.argmax().item()]
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), 
                          (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x), int(y) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()