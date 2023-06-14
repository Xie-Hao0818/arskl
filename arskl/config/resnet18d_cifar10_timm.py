var = dict(
    name='torch/cifar10',
    root='./dataset_img',
    input_size=32,
    num_classes=10,
    in_chans=3,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    batch_size=128,
    model_name='resnet18d',
    pretrained=True,
    epoch=50,
    optim='lion',
    auto_augment='rand-m6-n3-mstd0.5',
    num_workers=4,
    devices=[0],
    accumulate_grad_batches=1,
)
dateset = dict(
    type='TimmDataModule',
    traintf_cfg=dict(
        type='TimmTransform',
        input_size=var['input_size'],
        is_training=True,
        auto_augment=var['auto_augment'],
        mean=var['mean'],
        std=var['std'],
    ),
    valtf_cfg=dict(
        type='TimmTransform',
        input_size=var['input_size'],
        mean=var['mean'],
        std=var['std'],
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
)
