# -------------数据集---------------------
_base_ = ['./mnist1989.py']
# -------------服务器---------------------
pth_root = '/home/chenjunfen/workspace/XZH/arskl/best_ckpt/mnist'
log_root = '/home/chenjunfen/workspace/XZH/arskl/log/mnist'
teacher_root = ''
test = dict(
    is_test=False,
    ckpt_path='/home/chenjunfen/workspace/XZH/arskl/best_ckpt/mnist/epoch=21-vacc=97.4602-vloss=0.080.ckpt',
)
reload = dict(
    is_reload=False,
    ckpt_path='',
)
precision = 'bf16-mixed'
devices = [0, 1]
num_workers = 4
# --------------旋钮---------------------
batch_size = 32
accumulate_grad_batches = 1
var = dict(
    verbose=True,
    epoch=80,
    optim_type='Lion',
    lr=3e-4,
    weight_decay=5e-4,
    gradient_clip_val=40,
    compile=False,
    label_smoothing=0,
    distillation_type='none',
    enable_checkpointing=True,
    num_classes={{_base_.num_classes}},
)
model = dict(
    type='Net_1989_1',
)
# model = dict(
#     type='TimmModel',
#     model_name='resnet18d',
#     pretrained=True,
#     in_chans=1,
#     num_classes=10,
# )
scheduler = dict(
    type='ReduceLROnPlateau',
    mode='min',
    factor=0.5,
    patience=2,
    verbose=False,
    cooldown=1,
    min_lr=0,
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
    ckpt_path=teacher_root
)
# --------------结构------------------
dataset = dict(
    train_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers,
    ),
    valid_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers,
    ),
)
optim = dict(
    type=var['optim_type'],
    lr=var['lr'],
    weight_decay=var['weight_decay'],
)
learner = dict(
    type='Learner_89',
    model_cfg=model,
    optim_cfg=optim,
    scheduler_cfg=scheduler,
    hyper_cfg=var,
    teacher_cfg=teacher_cfg,
)
ckpt = dict(
    dirpath=pth_root,
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
    precision=precision,
    devices=devices,
    default_root_dir=log_root,
    accumulate_grad_batches=accumulate_grad_batches,
    enable_checkpointing=var['enable_checkpointing'],
    # strategy='ddp_find_unused_parameters_true',
    gradient_clip_val=var['gradient_clip_val'],
)
