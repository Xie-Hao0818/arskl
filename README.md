# arskl

基于ptyorch,lightning,timm,mmcv的计算机视觉算法模型库

### 优点
- 1.配置代码逻辑分离
- 2.可读性高，容易维护
- 3.方便追踪复现最新SOTA

### 训练
```bash
bash dist_train.sh {config}
bash /home/chenjunfen/workspace/XZH/arskl/tools/dist_train.sh /home/chenjunfen/workspace/XZH/arskl/configs/resnet18d_cifar10_timm.py
```