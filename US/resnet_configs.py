_base_ = [
    #'configs/_base_/models/resnet50.py',
    # 'configs/_base_/datasets/imagenet_bs256_rsb_a12.py',
    'configs/_base_/schedules/imagenet_bs2048_rsb.py',
    'configs/_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='weights/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth',
            prefix='backbone', ),
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2048,
        # loss=dict(
        #     type='LabelSmoothLoss',
        #     num_classes=2,
        #     label_smooth_val=0.2,
        #     mode='original',
        #     reduction='mean',
        #     use_sigmoid=True,
        #     loss_weight=1.0,),
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=(1,2)),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0,),
        topk=(1),),
    # train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)),
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
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=256),
    # dict(type='Resize', scale=256),
    dict(type='PackInputs'),
]

# 训练数据设置
train_dataloader = dict(
    batch_size=16,  # 每张 GPU 的 batchsize
    num_workers=2,  # 每个 GPU 的线程数
    dataset=dict(
        type='CustomDataset',
        data_prefix='/mnt/e/skx/Breast-Ultrasound/Datasets/Ultrasound/choose/MRI-t1/4folds/fold_0/train',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        with_label=True,  # or False for unsupervised tasks
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=16,  # 每张 GPU 的 batchsize
    num_workers=2,  # 每个 GPU 的线程数
    dataset=dict(
        type='CustomDataset',
        data_prefix='/mnt/e/skx/Breast-Ultrasound/Datasets/Ultrasound/choose/MRI-t1/4folds/fold_0/test',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        with_label=True,  # or False for unsupervised tasks
        pipeline=test_pipeline
    ))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Lamb', lr=1e-6, weight_decay=0.1))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
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
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3, save_best="single-label/f1-score", rule='greater'),)
    # visualization=dict(type='VisualizationHook', enable=True, interval=1,),)
    # save_best ['accuracy/topk-1',"single-label/precision","single-label/recall","single-label/f1-score"]
val_evaluator = [
    dict(type='Accuracy', topk=(1)),
    dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score']),
]
test_evaluator = val_evaluator

work_dir = './work_dirs/resnet50_MRI_50/fold_0'