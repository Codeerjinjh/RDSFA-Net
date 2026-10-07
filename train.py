from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')

#nohup python train.py --resume > train.log 2>&1 &
#tail -f train.log

model = RTDETR(r"improve_multimodal/rtdetr/rtdetr-l-midfusion-RDSFA-Pro.yaml")
model.train(
    data=r"datalvip.yaml",
    epochs=220,                      # ⭐ 200-300轮，RTDETR收敛慢
    imgsz=640,
    workers=8,
    batch=64,                         # ⭐ batch至少8，如果显存够上16更好
    device=[0],
    optimizer='AdamW',
    amp=True,                        # ⭐ 开混合精度，省显存+加速

    # ---------- 学习率 ----------
    lr0=0.0001,
    #lr0=0.0005,                      # AdamW 1e-4 正确
    lrf=0.001,                  #0.001      # 最终 lr = lr0 * lrf = 1e-6
    momentum=0.937,                  # Adam beta1=0.937 更好
    weight_decay=0.0001,             # AdamW weight_decay
    
    # ---------- warmup ----------
    warmup_epochs=5.0,               # ⭐ 增加到5轮
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # ---------- 损失权重 ----------
    box=7.5,
    cls=0.8,
    dfl=1.5,
    
    # ---------- 关键增强 ⭐ ----------
    hsv_h=0.015,                     # 色调增强
    hsv_s=0.7,                       # 饱和度增强
    hsv_v=0.4,                       # 明度增强
    translate=0.2,                   # ⭐ 平移从0.1→0.2
    scale=0.5,                       # 缩放
    fliplr=0.5,                      # 左右翻转
    
    # ⭐⭐⭐ 以下两个是核心提点增强，必须开！
    mosaic=1.0,                      # ⭐⭐⭐ Mosaic，RTDETR默认就是1.0
    mixup=0.2,                       # ⭐⭐⭐ Mixup，增加多样性防止过拟合
    
    # ---------- 正则化 ----------
    nbs=64,                          # 名义batch=64，梯度累积
    label_smoothing=0.05,             # 单类检测不需要
    copy_paste=0.0,
    auto_augment='randaugment',      # 分类的aug，检测可以保留
    erasing=0.0,                     # 检测不需要
    
    # ---------- 验证 ----------
    val=True,
    plots=True,                      # 保存曲线图

    project='runs/train/RDSFA-Pro', # 实验结果目录
    #name='LLVIP_exp',
    name='M3FD_exp',
)

