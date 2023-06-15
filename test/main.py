import matplotlib.pyplot as plt
import torch
from pyskl.datasets import build_dataloader
from pyskl.datasets.pose_dataset import PoseDataset
import timm
from einops import reduce, rearrange
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

ann_file = '/inspur/hdfs6/chenjunfen/xzh/dataset/hmdb51_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=64, num_clips=3),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(48, 48), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=64, num_clips=3),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(48, 48), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
EPOCH = 50
model = timm.create_model('vit_small_patch32_384.augreg_in21k_ft_in1k', num_classes=51, pretrained=True, in_chans=3,
                          drop_rate=0.4).cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
if __name__ == '__main__':
    # Train dataset
    dataset = PoseDataset(ann_file=ann_file, pipeline=train_pipeline, split='train1')
    dataloader_setting = dict(
        videos_per_gpu=128,
        workers_per_gpu=8,
        shuffle=True,
        seed=42)
    data_loader = build_dataloader(dataset, **dataloader_setting)
    # Valid dataset
    val_dataset = PoseDataset(ann_file=ann_file, pipeline=val_pipeline, split='test1')
    val_dataloader_setting = dict(
        videos_per_gpu=128,
        workers_per_gpu=8,
        seed=42)
    val_data_loader = build_dataloader(val_dataset, **val_dataloader_setting)
    for t in range(EPOCH):
        print(f"Epoch {t + 1}\n-------------------------------")
        # Train loop
        model.train()
        size = len(data_loader.dataset)
        for batch, item in enumerate(data_loader):
            imgs, label = item['imgs'].cuda(), item['label'].cuda()
            # print(label.shape)
            label = rearrange(label, 'n 1->n')
            # Compute prediction error
            predict = model(imgs)
            loss = loss_fn(predict, label)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Loss
            if batch % 7 == 0:
                loss, current = loss.item(), (batch + 1) * len(imgs)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # Valid loop
        model.eval()
        size = len(val_data_loader.dataset)
        num_batches = len(val_data_loader)
        test_loss, correct = 0, 0
        for item in val_data_loader:
            imgs, label = item['imgs'].cuda(), item['label'].cuda()  # torch.Size([4, 3, 224, 224])
            with torch.no_grad():
                predict = model(imgs)
                loss = loss_fn(predict, label)
                test_loss += loss.item()
                correct += (predict.argmax(1) == label).type(torch.float).sum().item()
        # Loss & Acc
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")

        scheduler.step()
