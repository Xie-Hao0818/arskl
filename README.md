# arskl

Office code for "DPConv3D: Dual Pathway Action Recognition Based on Skeleton Heatmap"

### Data



### Model
![双路径骨骼点热图动作识别架构图-第 10 页.png](..%2F..%2F..%2FDesktop%2F%CB%AB%C2%B7%BE%B6%B9%C7%F7%C0%B5%E3%C8%C8%CD%BC%B6%AF%D7%F7%CA%B6%B1%F0%BC%DC%B9%B9%CD%BC-%B5%DA%2010%20%D2%B3.png)

### Training
You can use the following command to train a model.
```bash
bash dist_train.sh ${CONFIG_FILE}
bash tools/dist_train.sh configs/4090x2/hmdb51/joint.py
```

### Testing
