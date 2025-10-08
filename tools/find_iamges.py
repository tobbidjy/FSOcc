from nuscenes.nuscenes import NuScenes

# 初始化nuScenes对象
dataset_version = 'v1.0-mini' # 修改数据集版本与路径
dataset_root = '/home/ubuntu/code/xuzeyuan/SurroundOcc/data/nuscenes/others/mini'
nusc = NuScenes(version=dataset_version, dataroot=dataset_root, verbose=True)

# 从点云文件名获取sample_token
lidar_filename = "n015-2018-08-02-17-16-37+0800__LIDAR_TOP__1533201470448696.pcd.bin"
sample_token = lidar_filename.split('__')[0]

# 获取样本数据
sample = nusc.get('sample', sample_token)

# 获取所有相机数据
camera_data = {}
for camera in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
    camera_token = sample['data'][camera]
    camera_sample = nusc.get('sample_data', camera_token)
    camera_data[camera] = camera_sample['filename']

print(camera_data)

# camera_data现在包含6个相机图像的路径

15353850921