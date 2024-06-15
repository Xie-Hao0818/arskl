# arskl

Office code for "DPConv3D: Dual Pathway Action Recognition Based on Skeleton Heatmap"

### Data



### Model
![dual_pathway.png](pic%2Fdual_pathway.png)

### Training
You can use the following command to train a model.
```bash
bash dist_train.sh ${CONFIG_FILE}
bash tools/dist_train.sh configs/4090x2/hmdb51/joint.py
```

### Testing
