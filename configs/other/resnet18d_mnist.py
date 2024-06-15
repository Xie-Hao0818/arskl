dateset = dict(
    type='MNISTDataModule',
    root='./dataset_img',
    batch_size=256,
    num_workers=16
)
model = dict(
    type='TimmModel',
    model_name='resnet18d',
    pretrained=True,
    num_classes=10,
    in_chans=1
)
epoch = 50
learner = dict(
    type='LearnerImg',
    model_cfg=model,
    epoch=epoch,
    num_classes=10,
    lr=3e-4
)
trainer = dict(
    type='Trainer',
    max_epochs=epoch,
    precision='16-mixed'
)
