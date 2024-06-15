in_channel = 1
num_classes = 10
batch_size = 16
num_workers = 4
dataset = dict(
    type='MNIST1989DataModule',
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
