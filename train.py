from ultralytics import YOLO

# Load the base model - you can use 'yolov8n.pt' for a lightweight model
model = YOLO('yolov8n.pt')  # 'n' = nano, fast & lightweight. Or use 'yolov8s.pt' for more accuracy.

# Train the model using your dataset
model.train(
    data='D:\VINYASA\FINAL Project\Pothole_Final\data.yaml',   # path to your YAML file
    epochs=50,                  # number of training epochs
    imgsz=640,                  # input image size
    batch=8,                    # adjust depending on your RAM
    project='runs/train',       # where to save results
    name='pothole_model'        # experiment name
)
