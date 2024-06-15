var = dict(
    name='torch/cifar10',
    root='/home/chenjunfen/workspace/XZH/arskl/dataset_img',  # 数据集绝对路径
    default_root_dir='/home/chenjunfen/workspace/XZH/arskl',  # 日志保存绝对路径
    input_size=32,
    num_classes=10,
    in_chans=3,
    batch_size=128,
    model_name='resnet50d',
    pretrained=True,  # 关闭clash进行权重下载
    epoch=20,
    optim='lion',
    lr=3e-4,
    weight_decay=5e-4,
    auto_augment='rand-m6-n3-mstd1',
    num_workers=4,
    devices=[0, 1],
    accumulate_grad_batches=1,
    seed=3407,
)
dateset = dict(
    type='TimmDataModule',
    traintf_cfg=dict(
        type='TimmTransform',
        input_size=var['input_size'],
        is_training=True,
        auto_augment=var['auto_augment'],
    ),
    valtf_cfg=dict(
        type='TimmTransform',
        input_size=var['input_size'],
    ),
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
    strategy='ddp_find_unused_parameters_true',
    default_root_dir=var['default_root_dir'],
)
