# -------------数据集---------------------
_base_ = ['./ntu120.py']
# -------------服务器---------------------
data_root = '/root/autodl-tmp/data/ntu120_hrnet.pkl'
pth_root = '/root/autodl-tmp/best_pth/ntu120_xsub'
log_root = '/root/tf-logs/ntu120'
teacher_root = '/root/autodl-tmp/ckpt/ntu120_xsub/joint.pth'

test = dict(
    is_test=True,
    ckpt_path='/root/autodl-tmp/best_pth/ntu120_xsub/epoch=22-vacc=0.8572-vloss=1.450.ckpt',
)
reload = dict(
    is_reload=False,
    ckpt_path='/root/autodl-tmp/best_pth/ntu120_xsub/epoch=1-vacc=0.8554-vloss=1.455.ckpt',
)
precision = '16-mixed'
devices = [0] if test['is_test'] else [0, 1]
num_workers = 12  # 12
# --------------旋钮---------------------
batch_size = 32  # 32
accumulate_grad_batches = 1
train_times = 1
var = dict(
    verbose=False,
    epoch=100,
    optim_type='Adam',
    lr=12e-4,  # 12e-4
    weight_decay=1e-4,  # 1e-4
    gradient_clip_val=40,
    compile=False,
    label_smoothing=0.1,
    distillation_type='soft',
    enable_checkpointing=True,
    num_classes={{_base_.num_classes}},
)
model = dict(
    type='DualPath4',
    ckpt_path=teacher_root,
    num_classes={{_base_.num_classes}},
)
scheduler1 = dict(
    type='CosineAnnealingLR',
    T_max=var['epoch'],
)
scheduler2 = dict(
    type='ReduceLROnPlateau',
    mode='max',
    factor=0.5,
    patience=2,
    verbose=False,
    cooldown=1,
    min_lr=0,
)
teacher_cfg = dict(
    type='ResNet3d',
    backbone_cfg=dict(
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2,),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)
    ),
    cls_head_cfg=dict(
        in_channels=512,
        num_classes={{_base_.num_classes}},
        dropout=0.5
    ),
    ckpt_path=teacher_root
)
# --------------结构------------------
dataset = dict(
    train_dataset_cfg=dict(
        dataset=dict(
            ann_file=data_root,
        ),
        times=train_times,
    ),
    valid_dataset_cfg=dict(
        ann_file=data_root,
    ),
    test_dataset_cfg=dict(
        ann_file=data_root,
    ),
    train_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers,
    ),
    valid_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers,
    ),
    test_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers // 2,
    ),
)
optim = dict(
    type=var['optim_type'],
    lr=var['lr'],
    weight_decay=var['weight_decay'],
)
learner = dict(
    type='Learner_Skl',
    model_cfg=model,
    optim_cfg=optim,
    scheduler_cfg=scheduler1,
    hyper_cfg=var,
    teacher_cfg=teacher_cfg,
)
ckpt = dict(
    dirpath=pth_root,
    filename='{epoch}-{vacc:.4f}-{vloss:.3f}',
    save_last=False,
    monitor='vacc',
    save_top_k=1,
    mode='max',
    every_n_epochs=1,
)
trainer = dict(
    type='Trainer',
    ckpt_cfg=ckpt,
    max_epochs=var['epoch'],
    precision=precision,
    devices=devices,
    default_root_dir=log_root,
    accumulate_grad_batches=accumulate_grad_batches,
    enable_checkpointing=var['enable_checkpointing'],
    strategy='ddp_find_unused_parameters_true',
    gradient_clip_val=var['gradient_clip_val'],
    benchmark=True,
    sync_batchnorm=True,
)
