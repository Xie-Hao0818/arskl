var = dict(
    name='torch/cifar10',
    data_root='/home/chenjunfen/workspace/Dataset',
    log_root='/home/chenjunfen/workspace/XZH/arskl/log/cifar10',
    ckpt_root='/home/chenjunfen/workspace/XZH/arskl/best_ckpt/cifar10',
    num_classes=10,
    batch_size=24,
    epoch=50,
    num_workers=8,
    devices=[0, 1],
    lr=3e-4,
    weight_decay=5e-5,
    gradient_clip_val=40,
)
test = dict(
    is_test=False,
    ckpt_path='/home/chenjunfen/workspace/XZH/arskl/best_ckpt/cifar10/epoch=26-vacc=0.9809-vloss=0.064.ckpt',
)
reload = dict(
    is_reload=False,
    ckpt_path='',
)
model = dict(
    type='TimmModel',
    model_name='resnet50d',
    pretrained=True,
    in_chans=3,
    num_classes=10,
)
train_transform = [
    dict(type='Resize', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandAugment', config_str='rand-m8-n4-mstd0.5', hparams={}),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    dict(type='RandomErasing'),
    dict(type='Cutout', n_holes=1, length=16),

]
valid_transform = [
    dict(type='Resize', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]
dataset = dict(
    type='TimmDataModule',
    train_transform_cfg=train_transform,
    valid_transform_cfg=valid_transform,
    train_dataset_cfg=dict(
        name=var['name'],
        root=var['data_root'],
        download=False,
        split='train',
    ),
    valid_dataset_cfg=dict(
        name=var['name'],
        root=var['data_root'],
        download=False,
        split='validation',
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
)
optim = dict(
    type='SophiaG',
    lr=var['lr'],
    weight_decay=var['weight_decay'],
)
scheduler = dict(
    type='CosineAnnealingLR',
    epoch=var['epoch'],
    times=1,
)
learner = dict(
    type='LearnerImg',
    model_cfg=model,
    optim_cfg=optim,
    scheduler_cfg=scheduler,
    hyper_cfg=var,
)
ckpt = dict(
    dirpath=var['ckpt_root'],
    filename='{epoch}-{vacc:.4f}-{vloss:.3f}',
    save_last=False,
    monitor='vloss',
    save_top_k=1,
    mode='min',
    every_n_epochs=1,
)
trainer = dict(
    type='Trainer',
    ckpt_cfg=ckpt,
    max_epochs=var['epoch'],
    precision='bf16-mixed',
    devices=var['devices'],
    default_root_dir=var['log_root'],
    enable_checkpointing=True,
    gradient_clip_val=var['gradient_clip_val'],
    benchmark=True,
)
