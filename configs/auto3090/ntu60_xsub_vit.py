var = dict(
    info='',
    verbose=False,
    work_dir='/root/tf-logs/ntu60',
    data_path='/root/autodl-tmp/project/arskl/dataset/ntu60_hrnet.pkl',
    data_split=['xsub_train', 'xsub_val'],
    train_times=1,
    in_channel=17,
    num_classes=60,
    batch_size=16,
    epoch=100,
    optim='Lion',
    lr=1e-4,
    weight_decay=5e-4,
    num_workers=8,
    devices=[0],
    accumulate_grad_batches=1,
    seed=3407,
    enable_checkpointing=True,
    gradient_clip_val=60,
)
model = dict(type='CCT3D')
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(48, 48), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(48, 48), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

# -------------------------------------------------------------------------------------------------------------------------------
dataset = dict(
    type='PoseDataLoaderX',
    train_dataset_cfg=dict(
        dataset=dict(
            type='PoseDataset',
            ann_file=var['data_path'],
            pipeline=train_pipeline,
            split=var['data_split'][0],
        ),
        times=var['train_times'],
    ),
    valid_dataset_cfg=dict(
        ann_file=var['data_path'],
        pipeline=val_pipeline,
        split=var['data_split'][1],
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
)
# scheduler = dict(
#     type='CosineAnnealingLR',
#     epoch=var['epoch'],
#     times=var['train_times'],
# )
scheduler = dict(
    type='OneCycleLR',
    total_steps=var['epoch'],
    max_lr=var['lr'],
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
    strategy='ddp_find_unused_parameters_true',
    enable_checkpointing=var['enable_checkpointing'],
    gradient_clip_val=var['gradient_clip_val'],
)
