_base_ = [
    # 'configs/_base_/models/vgg16bn.py',
    # 'configs/_base_/datasets/imagenet_bs32_pil_resize.py',
    'configs/_base_/schedules/imagenet_bs256.py',
    'configs/_base_/default_runtime.py',
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VGG',
        depth=16,
        norm_cfg=dict(type='BN'),
        num_classes=2),
    neck=None,
    head=dict(
        type='ClsHead',

        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0),
        topk=(1),
    init_cfg=dict(
            type='Pretrained',
            checkpoint='weights/vgg/vgg16_bn_batch256_imagenet_20210208-7e55cd29.pth', ),
    ))



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
    # dict(type='RandomResizedCrop', scale=224),
    dict(type='Resize', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='EfficientNetCenterCrop', crop_size=256),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    # dict(type='Resize', scale=256),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

# 训练数据设置
train_dataloader = dict(
    batch_size=16,  # 每张 GPU 的 batchsize
    num_workers=2,  # 每个 GPU 的线程数
    dataset=dict(
        type='CustomDataset',
        data_prefix='/mnt/e/skx/Breast-Ultrasound/Datasets/cut/no_background_min/ALL',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        with_label=True,  # or False for unsupervised tasks
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=16,  # 每张 GPU 的 batchsize
    num_workers=2,  # 每个 GPU 的线程数
    dataset=dict(
        type='CustomDataset',
        data_prefix='/mnt/e/skx/Breast-Ultrasound/Datasets/cut/no_background_min/ALL',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        with_label=True,  # or False for unsupervised tasks
        pipeline=test_pipeline
    ))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.00002, momentum=0.9,  weight_decay=0.0001),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),)

# learning policy
param_scheduler = [
    # # warm up learning rate scheduler
    # dict(
    #     type='LinearLR',
    #     start_factor=0.0001,
    #     by_epoch=True,
    #     begin=0,
    #     end=5,
    #     # update by iter
    #     convert_to_iter_based=True),
    # # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=0,
        end=100)
    # dict(
    #     type='MultiStepLR',
    #     by_epoch=True,
    #     milestones=[30, 60, 90],
    #     gamma=0.5)
]
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100,)


default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3, save_best="auto"),)
    # visualization=dict(type='VisualizationHook', enable=True, interval=1,),)

val_evaluator = [
    dict(type='Accuracy', topk=(1)),
    dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score']),
]
test_evaluator = val_evaluator

work_dir = './work_dirs/vgg-cut'