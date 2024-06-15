var = dict(
    log_root='/root/tf-logs/ntu60',
    data_root='/root/autodl-tmp/arskl/data/ntu60_hrnet.pkl',
    ckpt_root='/root/autodl-tmp/arskl/best_pth/ntu60',
    data_split=['xsub_train', 'xsub_val'],
    train_times=1,
    in_channel=17,
    num_classes=60,
    batch_size=16,
    epoch=50,
    num_workers=4,
    optim_type='Lion',
    lr=3e-4,
    weight_decay=3e-4,
    devices=[0, 1, 2, 3],
    accumulate_grad_batches=1,
    gradient_clip_val=40,
    compile=False,
    label_smoothing=0.1,
    distillation_type='none',
)
test = dict(
    is_test=False,
    ckpt_path='/home/chenjunfen/workspace/XZH/arskl/best_ckpt/ntu60/epoch=25-vacc=0.9003-vloss=0.374.ckpt',
)
reload = dict(
    is_reload=False,
    ckpt_path='/root/autodl-tmp/arskl/best_pth/ntu60/epoch=25-vacc=0.9066-vloss=1.034.ckpt',
)
model = dict(
    type='ResNet3D_Diy',
)
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=32, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),  # 64
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label']),
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=32, num_clips=3),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),  # 64
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs']),
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=49, num_clips=5),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(32, 32), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs']),
    dict(type='Vedio2Img')
]
# -------------------------------------------------------------------------------------------------------------------------------
dataset = dict(
    type='PoseDataLoaderX',
    train_dataset_cfg=dict(
        dataset=dict(
            type='PoseDataset',
            ann_file=var['data_root'],
            pipeline=train_pipeline,
            split=var['data_split'][0],
        ),
        times=var['train_times'],
    ),
    valid_dataset_cfg=dict(
        ann_file=var['data_root'],
        pipeline=val_pipeline,
        split=var['data_split'][1],
    ),
    test_dataset_cfg=dict(
        ann_file=var['data_root'],
        pipeline=test_pipeline,
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
        shuffle=True,
        pin_memory=True,
    ),
    test_dataloader_cfg=dict(
        batch_size=16,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
    ),
)
optim = dict(
    type=var['optim_type'],
    lr=var['lr'],
    weight_decay=var['weight_decay'],
)
scheduler = dict(
    type='CosineAnnealingLR',
    epoch=var['epoch'],
)

learner = dict(
    type='Learner_Skl',
    model_cfg=model,
    optim_cfg=optim,
    scheduler_cfg=scheduler,
    hyper_cfg=var,
)
ckpt = dict(
    dirpath=var['ckpt_root'],
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
    # precision='bf16-mixed',
    precision='16-mixed',
    devices=var['devices'],
    accumulate_grad_batches=var['accumulate_grad_batches'],
    default_root_dir=var['log_root'],
    enable_checkpointing=True,
    strategy='ddp_find_unused_parameters_true',
    gradient_clip_val=var['gradient_clip_val'],
)
