import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('runs/train/RDSFA-Pro/DVTOD_Pro/weights/best.pt')
    model.val(data='datadvtod.yaml',
                imgsz=640,
                batch=64,
                split='val',
                workers=8,
                device='0',
                )
 
