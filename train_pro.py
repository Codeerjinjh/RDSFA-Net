"""
RDSFA-Net Pro (~127M) Training Script
Usage:
 tmux at -t  jjh
 source ~/anaconda3/bin/activate
    python train_pro.py --data m3fd      # M3FD训练
    python train_pro.py --data dvtod     # DVTOD训练
    python train_pro.py --data llvip    # LLVIP训练
    torchrun --nproc_per_node=2 train_pro.py --data llvip
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_pro.py --data llvip --wandb disabled
"""
from ultralytics import RTDETR
import warnings, argparse
warnings.filterwarnings('ignore')

# 关闭 wandb 日志
import os
# 全局关闭wandb，优先级最高
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='m3fd', choices=['m3fd', 'dvtod', 'llvip'])
args = parser.parse_args()

datasets = {
    'm3fd': {
        'yaml': 'datam3fd.yaml', 'batch': 12, 'imgsz': 640,
        'name': 'M3FD_Pro', 'project': 'runs_train_RDSFA-Pro',  # 把 / 换成 _
    },
    'dvtod': {
        'yaml': 'datadvtod.yaml', 'batch': 16, 'imgsz': 640,
        'name': 'DVTOD_Pro', 'project': 'runs_train_RDSFA-Pro',
    },
    'llvip': {
        'yaml': 'datalvip.yaml', 'batch': 12, 'imgsz': 640,
        'name': 'LLVIP_Pro', 'project': 'runs_train_RDSFA-Pro',
    },

}
ds = datasets[args.data]

model = RTDETR("improve_multimodal/rtdetr/rtdetr-l-midfusion-RDSFA-Pro.yaml")

model.train(
    data=ds['yaml'],
    epochs=300,
    imgsz=ds['imgsz'],
    batch=ds['batch'],  
    device=[0,1],
    optimizer='AdamW',
    amp=False,

    # ⭐ 关闭多尺度 (DVTOD未对齐数据+大模型=巨慢+OOM)
    multi_scale=False,
    # 学习率
    lr0=0.0001,
    lrf=0.01,
    cos_lr=True,
    momentum=0.937,
    weight_decay=0.0001,

    warmup_epochs=5.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    box=7.5,
    cls=0.8,
    dfl=1.5,

    # 数据增强
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.2,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.5,
    close_mosaic=15,

    nbs=64,
    label_smoothing=0.1,
    auto_augment='randaugment',

    val=True,
    plots=True,
    save=True,
  

    project=ds['project'],
    name=ds['name'],
    exist_ok=True,
)
