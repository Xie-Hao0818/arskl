var = dict(
    info='',
    name='torch/cifar10',
    root='/home/chenjunfen/workspace/XZH/arskl/dataset_img',  # 数据集绝对路径
    default_root_dir='/home/chenjunfen/workspace/XZH/arskl',  # 日志保存绝对路径
    input_size=32,
    num_classes=10,
    in_chans=3,
    batch_size=64,
    epoch=50,
    optim='lion',
    lr=5e-6,  # finetune需要减少10-100倍学习率
    weight_decay=5e-4,
    num_workers=4,
    devices=[1],
    accumulate_grad_batches=1,
    seed=3407,
    enable_checkpointing=True,
)
traintf_cfg = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='Resize', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandAugment', config_str='rand-m8-n4-mstd0.5', hparams={}),  # rand-m8-n4-mstd0.5
    # dict(type='AutoAugment', config_str='original-mstd0.5', hparams={}),
    # dict(type='MixAugment', config_str='augmix-m5-w4-d2', hparams={}),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    dict(type='RandomErasing'),
    dict(type='Cutout', n_holes=1, length=16),

]
valtf_cfg = [
    dict(type='Resize', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]
dateset = dict(
    type='TimmDataModule',
    traintf_cfg=traintf_cfg,
    valtf_cfg=valtf_cfg,
    trainds_cfg=dict(
        name=var['name'],
        root=var['root'],
        download=True,
        split='train',
    ),
    valds_cfg=dict(
        name=var['name'],
        root=var['root'],
        download=True,
        split='validation',
    ),
    traindl_cfg=dict(
        batch_size=var['batch_size'],
        num_workers=var['num_workers'],
        shuffle=True,
        pin_memory=True,
    ),
    valdl_cfg=dict(
        batch_size=var['batch_size'],
        num_workers=var['num_workers'],
        shuffle=False,
        pin_memory=True,
    ),
)
model = dict(
    type='HFModel',
    path='aaraki/vit-base-patch16-224-in21k-finetuned-cifar10'
)
optim = dict(
    opt=var['optim'],
    lr=var['lr'],
    weight_decay=var['weight_decay'],
)
learner = dict(
    type='LearnerImg',
    model_cfg=model,
    optim_cfg=optim,
    hyper_cfg=var,
    epoch=var['epoch']
)
trainer = dict(
    type='Trainer',
    max_epochs=var['epoch'],
    precision='16-mixed',
    devices=var['devices'],
    accumulate_grad_batches=var['accumulate_grad_batches'],
    # strategy='ddp_find_unused_parameters_true',
    default_root_dir=var['default_root_dir'],
    enable_checkpointing=var['enable_checkpointing'],
)
