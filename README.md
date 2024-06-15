# arskl

Office code for "DPConv3D: Dual Pathway Action Recognition Based on Skeleton Heatmap"

### 优点
- 1.配置代码逻辑分离
- 2.可读性高，容易维护
- 3.方便追踪复现最新SOTA

### 训练
```bash
bash dist_train.sh {config}
bash /home/chenjunfen/workspace/XZH/arskl/tools/dist_train.sh /home/chenjunfen/workspace/XZH/arskl/configs/resnet18d_cifar10_timm.py
```