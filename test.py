from pyskl.datasets import build_dataloader
from pyskl.datasets.pose_dataset import PoseDataset
from einops import rearrange, reduce
import matplotlib.pyplot as plt

ann_file = '/inspur/hdfs6/chenjunfen/xzh/data/hmdb51_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=49, num_clips=3),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 72)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=49, num_clips=3),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(32, 32), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

# Train data
dataset = PoseDataset(ann_file=ann_file, pipeline=train_pipeline, split='train1')
dataloader_setting = dict(
    videos_per_gpu=10,
    workers_per_gpu=8,
    shuffle=True,
    seed=42)
data_loader = build_dataloader(dataset, **dataloader_setting)
# Valid data
val_dataset = PoseDataset(ann_file=ann_file, pipeline=val_pipeline, split='test1')
val_dataloader_setting = dict(
    videos_per_gpu=10,
    workers_per_gpu=8,
    seed=42)
val_data_loader = build_dataloader(val_dataset, **val_dataloader_setting)

for idx, item in enumerate(data_loader):
    if (idx == 1):
        break
    imgs = item['imgs']
    print(imgs.shape)  # torch.Size([10, 3, 17, 49, 32, 32])
    imgs = reduce(imgs, 'b n c t h w->b n c h w', 'max')
    plt.imshow(imgs[0][0][0])
    plt.show()
