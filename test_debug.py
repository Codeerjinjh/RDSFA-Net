import sys
import traceback
import torch

# 先测试模型是否能正常构建
from ultralytics import RTDETR

try:
    model = RTDETR(r"improve_multimodal/rtdetr/rtdetr-l-midfusion-RDSFA-lite.yaml")
    print("Model built successfully!")
    
    # 模拟6通道输入测试前向
    dummy = torch.randn(1, 6, 640, 640)
    print(f"Input shape: {dummy.shape}")
    out = model(dummy)
    if isinstance(out, list):
        print(f"Output: {[o.shape for o in out]}")
    else:
        print(f"Output shape: {out.shape}")
        
except Exception as e:
    traceback.print_exc()
