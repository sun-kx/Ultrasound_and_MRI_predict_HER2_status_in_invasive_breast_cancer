_base_ = [
    'configs/_base_/models/convnext_v2/base.py',
    'configs/_base_/datasets/imagenet_bs64_swin_384_cls.py',
    'configs/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    'configs/_base_/default_runtime.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth',
            # checkpoint='/home/hdd0/skx/tct/mmpretrain/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-d8579f84.pth',
            checkpoint='weights/convnext-v2/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth',
            prefix='backbone',),
    ),
    head=dict(
        type='LinearClsHead',
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.2),
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=(1, 2)),
        num_classes=2,
    ),
)

bgr_mean = [103.53, 116.28, 123.675]
data_preprocessor = dict(
    num_classes=2,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=False
    )

# dataset settings
# dataset_type = 'CustomDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

# 训练数据设置
train_dataloader = dict(
    batch_size=4,  # 每张 GPU 的 batchsize
    num_workers=2,  # 每个 GPU 的线程数
    dataset=dict(
        type='CustomDataset',
        data_prefix='/mnt/e/skx/Breast-Ultrasound/Datasets/Ultrasound/choose/MRI-t1/4folds/fold_3/train',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        with_label=True,  # or False for unsupervised tasks
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=4,  # 每张 GPU 的 batchsize
    num_workers=2,  # 每个 GPU 的线程数
    dataset=dict(
        type='CustomDataset',
        data_prefix='/mnt/e/skx/Breast-Ultrasound/Datasets/Ultrasound/choose/MRI-t1/4folds/fold_3/test',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        with_label=True,  # or False for unsupervised tasks
        pipeline=test_pipeline
    ))
test_dataloader = val_dataloader


# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=2e-5),
    clip_grad=None,
)

# learning policy
param_scheduler = [

    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min=3.0e-6,
        by_epoch=True,
        begin=0,
        end=100)

]
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100,)


default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3, save_best="single-label/f1-score", rule='greater'),)
    # visualization=dict(type='VisualizationHook', enable=True, interval=1,),)
    # save_best ['accuracy/topk-1',"single-label/precision","single-label/recall","single-label/f1-score"]
val_evaluator = [
    dict(type='Accuracy', topk=(1)),
    dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score']),
]
test_evaluator = val_evaluator

work_dir = './work_dirs/convnext_v2_MRI_50_1/fold_2'