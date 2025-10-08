# Inherit base config
_base_ = ['./surroundocc_r50.py']

# Modify training pipeline to add MGM
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False),

    # Add MGM augmentation
    dict(
        type='MultiViewGridMask',
        use_h=True,
        use_w=True,
        rotate=1,
        offset=False,
        ratio_range=(0.5, 0.8),
        density_range=(0.4, 0.7),
        mode=1,
        prob_schedule='dynamic',  # or 'fixed'
        edge_margin=50,
        small_obj_threshold=0.1,
        large_mask_ratio=0.05,
        detector_path='yolov8s.pt',  # Path to YOLOv8 weights
        cache_dir='./work_dirs/mgm_cache'
    ),

    dict(type='RandomFlip3D'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_occ'])
]

# Update dataset config
data = dict(
    train=dict(pipeline=train_pipeline)
)