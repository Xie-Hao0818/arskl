in_channel = 1
num_classes = 10
batch_size = 16
num_workers = 4
dataset = dict(
    type='TimmDataModule',
    train_transform_cfg=dict(
        type='TimmTransform',
        input_size=28,
        is_training=True,
        auto_augment='rand-m6-n3-mstd1',
        mean=(0.1308,),
        std=(0.3082,),
    ),
    valid_transform_cfg=dict(
        type='TimmTransform',
        input_size=28,
        mean=(0.1308,),
        std=(0.3082,),
    ),
    train_dataset_cfg=dict(
        name='torch/mnist',
        root='./dataset_img',
        download=True,
        split='train',
    ),
    valid_dataset_cfg=dict(
        name='torch/mnist',
        root='./dataset_img',
        download=True,
        split='validation',
    ),
    train_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    ),
    valid_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    ),
)
