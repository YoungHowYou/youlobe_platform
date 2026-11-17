ROOT = 'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young'
crop_size = (
    512,
    512,
)
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'scratchseg',
        'mmseg.visualization',
    ])
dataset_common = dict(
    ann_dir=
    'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\tr_data\\masks_binary',
    img_dir=
    'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\tr_data\\images',
    patch_size=512,
    pipeline=[
        dict(type='PackSegInputs'),
    ],
    type='DefectPatchDataset')
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=500, save_best='mIoU', type='CheckpointHook'),
    logger=dict(interval=20, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(pretrained=True, type='ResUNet34Backbone'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        channels=64,
        in_channels=64,
        in_index=0,
        loss_decode=dict(
            cldice_weight=1.0,
            loss_weight=1.0,
            pos_weight=5.0,
            type='BCEDiceclDiceLoss'),
        num_classes=1,
        num_convs=2,
        type='FCNHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
optim_wrapper = dict(
    accumulative_counts=4,
    loss_scale='dynamic',
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            enc0=dict(lr_mult=0.1),
            enc1=dict(lr_mult=0.1),
            enc2=dict(lr_mult=0.1),
            enc3=dict(lr_mult=0.1),
            enc4=dict(lr_mult=0.1))),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=500,
        by_epoch=False,
        end=8000,
        eta_min=1e-06,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_dir=
        'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\tr_data\\masks_binary',
        img_dir=
        'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\tr_data\\images',
        is_train=False,
        patch_size=512,
        patches_per_image=1,
        pipeline=[
            dict(type='PackSegInputs'),
        ],
        split_file=
        'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\scratch_seg\\data\\splits\\val.txt',
        type='DefectPatchDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=8000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_dir=
        'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\tr_data\\masks_binary',
        img_dir=
        'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\tr_data\\images',
        is_train=True,
        patch_size=512,
        patches_per_image=4,
        pipeline=[
            dict(type='PackSegInputs'),
        ],
        split_file=
        'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\scratch_seg\\data\\splits\\train.txt',
        type='DefectPatchDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_dir=
        'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\tr_data\\masks_binary',
        img_dir=
        'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\tr_data\\images',
        is_train=False,
        patch_size=512,
        patches_per_image=1,
        pipeline=[
            dict(type='PackSegInputs'),
        ],
        split_file=
        'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\scratch_seg\\data\\splits\\val.txt',
        type='DefectPatchDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'c:\\Users\\haoyu.yang\\Desktop\\source\\MMSegmentation_young\\scratch_seg\\work_dirs\\resunet34_512_e_best'
