"""
RDSFA-Net Pro 验证脚本
用法: python val_pro.py
"""
from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')

model_path = "/home/s-zhangzy/jinjh/yolo-multimodal/runs_train_RDSFA-Pro/M3FD_Pro/weights/best.pt"

model = RTDETR(model_path)

# val集验证
metrics = model.val(
    data="datam3fd.yaml",
    batch=16,
    imgsz=640,
    device=[0],
    project="runs_train_RDSFA-Pro_M3FD_Pro",
    name="val_results",
    exist_ok=True,
    plots=True,        # 生成PR曲线等图
    save_json=True,    # 保存检测结果json
    conf=0.001,        # 低阈值保证召回率
    iou=0.7,
    max_det=300,
    half=False,        # 用FP32保持精度
    verbose=True,
)

# 输出详细结果
print("\n" + "="*60)
print("📊 M3FD 验证结果")
print("="*60)
print(f"  mAP50:    {metrics.box.map50:.4f}")
print(f"  mAP50-95: {metrics.box.map:.4f}")
print(f"  Precision: {metrics.box.mp:.4f}")
print(f"  Recall:    {metrics.box.mr:.4f}")
print("="*60)

# 每类结果
if hasattr(metrics.box, 'ap_class_index') and hasattr(metrics.box, 'ap'):
    names = {0: 'person', 1: 'car', 2: 'bicycle'}
    print("\n📊 每类AP结果:")
    print(f"  {'Class':<12} {'AP50':<10} {'AP75':<10}")
    print("  " + "-"*32)
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        ap50 = metrics.box.ap[i][0]
        ap75 = metrics.box.ap[i][1]
        print(f"  {names.get(cls_idx, f'class{cls_idx}'):<12} {ap50:<10.4f} {ap75:<10.4f}")

# 保存结果到文件
with open("runs/train/RDSFA-Pro/M3FD_Pro/val_results.txt", 'w') as f:
    f.write("M3FD Validation Results\n")
    f.write(f"mAP50: {metrics.box.map50:.4f}\n")
    f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
    f.write(f"Precision: {metrics.box.mp:.4f}\n")
    f.write(f"Recall: {metrics.box.mr:.4f}\n")

print(f"\n✅ Validation complete! Results saved to val_results.txt")
