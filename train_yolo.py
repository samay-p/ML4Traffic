from ultralytics import YOLO

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    model = YOLO("yolov8n.pt")  # YOLOv8
    
    model.train(
        data="data.yaml",  
        epochs=1000,       # Reduced due to small dataset
        imgsz=640,         
        batch=8,           # Smaller batch for 6GB GPU
        workers=0,         # Windows fix
        amp=False,         # GTX 1660 Ti compatibility
        patience=30,       # Early stopping
        augment=True,      # Important since small dataset
        hsv_h=0.015,       # Color augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,       # Rotation
        translate=0.2,     # Translation
        scale=0.5,         # Scaling
        fliplr=0.5,        # Horizontal flip
        mosaic=1.0,        # Mosaic augmentation
    )
    
    print("Training done.")
