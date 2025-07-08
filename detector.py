from ultralytics import YOLO
import cv2
import os

model = YOLO('yolov8n.pt')  

vehicle_classes = ['car', 'truck', 'motorcycle']


input_dir = 'input_images'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

summary = {}

for image_file in os.listdir(input_dir):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)

        results = model(image_path)

        count = {'car': 0, 'truck': 0, 'motorcycle': 0}

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                if label in vehicle_classes and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    count[label] += 1

        out_path = os.path.join(output_dir, image_file)
        cv2.imwrite(out_path, image)

        summary[image_file] = count

print("Vehicle Count Summary:")
for name, count in summary.items():
    print(f"{name}: {count}")
