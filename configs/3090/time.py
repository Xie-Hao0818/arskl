var = dict(
    info='',
    verbose=False,
    work_dir='/home/chenjunfen/workspace/XZH/arskl/time_workspace',
    train_times=1,
    epoch=10,
    optim='Adam',
    lr=1e-4,
    devices=[0, 1],  # train,valid
    # devices=[1],  # predict
    accumulate_grad_batches=1,
    seed=3407,
    enable_checkpointing=True,
    seq_len=336,
    pred_len=672,
    enc_in=1,
)
test = dict(
    is_test=True,
    ckpt_path='/home/chenjunfen/workspace/XZH/arskl/best_pth/time/power-vloss=0.148.ckpt',
)
predict = dict(
    is_predict=False,
    ckpt_path='/home/chenjunfen/workspace/XZH/arskl/best_pth/time/epoch=2-vloss=0.148.ckpt',
)
model = dict(
    type='Linear',
    seq_len=var['seq_len'],
    pred_len=var['pred_len'],
    individual=False,
    enc_in=var['enc_in'],
)
# -------------------------------------------------------------------------------------------------------------------------------
dataset = dict(
    type='TimeDataLoader',

)
optim = dict(
    type=var['optim'],
    lr=var['lr'],
)
scheduler_c = dict(
    type='CosineAnnealingLR',
    epoch=var['epoch'],
    times=var['train_times'],
)
scheduler_o = dict(
    type='OneCycleLR',
    total_steps=var['epoch'],
    max_lr=var['lr'],
)

learner = dict(
    type='LearnerByTime',
    model_cfg=model,
    optim_cfg=optim,
    scheduler_cfg=scheduler_c,
    hyper_cfg=var,
)
lr_finder = dict(
    min_lr=1e-8,
    max_lr=1,
    num_training_steps=100,
    mode='exponential',
    early_stop_threshold=4,
    attr_name='lr'
)
ckpt = dict(
    dirpath='/home/chenjunfen/workspace/XZH/arskl/best_pth/time',
    filename='{epoch}-{vloss:.3f}',
    save_last=False,
    monitor='vloss',
    save_top_k=1,
    mode='min',
    every_n_epochs=1,
)
trainer = dict(
    type='Trainer',
    ckpt_cfg=ckpt,
    lr_finder=lr_finder,
    max_epochs=var['epoch'],
    # precision='bf16-mixed',
    devices=var['devices'],
    accumulate_grad_batches=var['accumulate_grad_batches'],
    default_root_dir=var['work_dir'],
    enable_checkpointing=var['enable_checkpointing'],
)
