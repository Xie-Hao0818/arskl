var = dict(
    verbose=False,
    log_root='/home/chenjunfen/workspace/XZH/arskl/log/ntu60',
    data_root='/home/chenjunfen/workspace/XZH/dataset/pyskl/ntu60_hrnet.pkl',
    ckpt_root='/home/chenjunfen/workspace/XZH/arskl/best_ckpt/ntu60',
    data_split=['xsub_train', 'xsub_val'],
    train_times=1,
    in_channel=17,
    num_classes=60,
    batch_size=32,
    epoch=300,
    num_workers=4,
    optim_type='Lion',
    lr=3e-4,
    weight_decay=3e-4,
    devices=[1],
    accumulate_grad_batches=2,
    gradient_clip_val=40,
    compile=False,
    label_smoothing=0,
    distillation_type='none',
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
        num_classes=60,
        dropout=0.5
    ),
    ckpt_path='/home/chenjunfen/workspace/XZH/arskl/ckpt/ntu60_xsub/joint.pth',
)
test = dict(
    is_test=False,
    ckpt_path='/home/chenjunfen/workspace/XZH/arskl/best_ckpt/ntu60/epoch=25-vacc=0.9003-vloss=0.374.ckpt',
)
reload = dict(
    is_reload=False,
    ckpt_path='/home/chenjunfen/workspace/XZH/arskl/best_ckpt/ntu60/epoch=13-vacc=0.8792-vloss=0.388.ckpt',
)
model = dict(
    type='ResNet3d_Skl_NTU60',

)
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label']),
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs']),
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs']),
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
        num_workers=4,
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
    T_max=var['epoch'],
)
learner = dict(
    type='Learner_Skl',
    model_cfg=model,
    optim_cfg=optim,
    scheduler_cfg=scheduler,
    hyper_cfg=var,
    teacher_cfg=teacher_cfg,
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
    # strategy='ddp_find_unused_parameters_true',
    gradient_clip_val=var['gradient_clip_val'],
    benchmark=True,
    sync_batchnorm=True,
)
