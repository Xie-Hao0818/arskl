var = dict(
    name='torch/cifar10',
    root='/home/chenjunfen/workspace/XZH/arskl/dataset_img',  # 数据集绝对路径
    default_root_dir='/home/chenjunfen/workspace/XZH/arskl',  # 日志保存绝对路径
    input_size=32,
    num_classes=10,
    in_chans=3,
    batch_size=128,
    model_name='resnet18d',
    pretrained=True,  # 关闭clash进行权重下载
    epoch=50,
    optim='lion',
    auto_augment='rand-m6-n3-mstd1',
    num_workers=4,
    devices=[0, 1],
    accumulate_grad_batches=1,
    seed=3407,
)
traintf_cfg = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    dict(type='Cutout', n_holes=1, length=16),
]
valtf_cfg = [
    dict(type='ToTensor'),
    dict(type='Normalize', mean=(0.4942, 0.4851, 0.4504), std=(0.2467, 0.2429, 0.2616)),
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
    type='TimmModel',
    model_name=var['model_name'],
    pretrained=var['pretrained'],
    num_classes=var['num_classes'],
    in_chans=var['in_chans'],
)
optim = dict(
    opt=var['optim'],
    lr=3e-4,
    weight_decay=5e-4,
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
    strategy='ddp_find_unused_parameters_true',
    default_root_dir=var['default_root_dir'],
)
