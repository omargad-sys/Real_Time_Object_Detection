import cv2
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    model = YOLO('yolov8m.pt')

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = cv2.flip(frame, 1) # Flip horizontally

        # Perform object detection
        results = model(frame, stream=True)

        # Process results and draw bounding boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                label = model.names[cls]

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('YOLOv8 Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
