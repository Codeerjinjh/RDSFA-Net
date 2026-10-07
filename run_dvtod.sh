#!/bin/bash
cd /root/autodl-tmp/yolo-multimodal
python3 train_pro.py --data dvtod > runs/train/RDSFA-Pro/DVTOD_Pro.log 2>&1
