var = dict(
    info='TubeViT X PoseConv3D',
    verbose=False,
    work_dir='/root/tf-logs/hmdb51',
    data_root='/root/autodl-tmp/project/arskl/dataset/hmdb51_pose_split1.pkl',
    video_shape=[17, 64, 48, 48],
    num_classes=51,
    batch_size=16,
    epoch=50,
    optim='Lion',
    lr=1e-5,
    weight_decay=5e-4,
    num_workers=8,
    devices=[0],
    accumulate_grad_batches=1,
    seed=3407,
    enable_checkpointing=True,
)
model = dict(
    type='TubeViT',
    num_classes=var['num_classes'],
    video_shape=var['video_shape'],
    num_layers=8,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=512,
)
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_transform = [
    dict(type='UniformSampleFrames', clip_len=64),
    dict(type='PoseDecode'),
    dict(type='PoseCompact'),
    dict(type='PoseResize', scale=(-1, 64)),
    dict(type='PoseRandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='PoseResize', scale=(48, 48), keep_ratio=False),
    dict(type='PoseFlip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=False, with_limb=True),
    dict(type='FormatShape', input_format='c t h w'),
    # dict(type='FramesToImgs1C'),
    dict(type='Collect', keys=['imgs', 'label']),
    dict(type='ImgsToTensor')
]
valid_transform = [
    dict(type='UniformSampleFrames', clip_len=64),
    dict(type='PoseDecode'),
    dict(type='PoseCompact'),
    dict(type='PoseResize', scale=(48, 48), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=False, with_limb=True),
    dict(type='FormatShape', input_format='c t h w'),
    # dict(type='FramesToImgs1C'),
    dict(type='Collect', keys=['imgs', 'label']),
    dict(type='ImgsToTensor')
]
dateset = dict(
    type='PoseDataModule',
    train_transform_cfg=train_transform,
    valid_transform_cfg=valid_transform,
    train_dataset_cfg=dict(
        root=var['data_root'],
        train=True,
    ),
    valid_dataset_cfg=dict(
        root=var['data_root'],
        train=False,
    ),
    train_dataloader_cfg=dict(
        batch_size=var['batch_size'],
        num_workers=var['num_workers'],
        shuffle=True,
        pin_memory=True,
    ),
    valid_dataloader_cfg=dict(
        batch_size=var['batch_size'],
        num_workers=var['num_workers'],
        shuffle=False,
        pin_memory=True,
    ),
)

optim = dict(
    type=var['optim'],
    lr=var['lr'],
    weight_decay=var['weight_decay'],
    # momentum=0.9,
)
scheduler = dict(
    type='CosineAnnealingLR',
    T_max=var['epoch'],
)
learner = dict(
    type='LearnerImg',
    model_cfg=model,
    optim_cfg=optim,
    scheduler_cfg=scheduler,
    hyper_cfg=var,
)
trainer = dict(
    type='Trainer',
    max_epochs=var['epoch'],
    precision='bf16-mixed',  # 适合3090GPU
    devices=var['devices'],
    accumulate_grad_batches=var['accumulate_grad_batches'],
    default_root_dir=var['work_dir'],
    enable_checkpointing=var['enable_checkpointing'],
    gradient_clip_val=0.5,  # 有效，可以避免梯度爆炸和提高模型精度
)
