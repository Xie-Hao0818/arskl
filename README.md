# arskl

Office code for "DPConv3D: Dual Pathway Action Recognition Based on Skeleton Heatmap"

### Download the pre-processed skeletons
We provide links to the pre-processed skeleton annotations, you can directly download them and use them for training & testing.
- [HMDB51 [2D Skeleton]](https://download.openmmlab.com/mmaction/pyskl/data/hmdb51/hmdb51_hrnet.pkl)
- [NTURGB+D [2D Skeleton]](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl)
- [NTURGB+D 120 [2D Skeleton]](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl)


### Training & Testing
You can use the following command to train or test a model.
```bash
bash dist_run.sh ${CONFIG_FILE}
bash tools/dist_run.sh configs/4090x2/hmdb51/joint.py
```
