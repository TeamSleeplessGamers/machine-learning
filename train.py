
from ultralytics import YOLO

# define model
model = YOLO("yolo11n.pt")

# Train the model
model.train(data="game-config.yaml", epochs=1000, imgsz=640,batch = 8, patience=0)