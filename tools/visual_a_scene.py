# 可视化一个场景，包括点云分割与占据预测结果


import shutil
import os, sys
import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
import torch
from PIL import Image
from pyquaternion.quaternion import Quaternion
import math


#-----------------------------------------------------------------------------#
colors = np.array(
    [
        [0, 0, 0, 255],        # 0 background          write
        [255, 120, 50, 255],  # 1 barrier              orangey
        [255, 192, 203, 255],  # 2 bicycle              pink
        [255, 255, 0, 255],  # 3 bus                  yellow
        [0, 150, 245, 255],  # 4 car                  blue
        [0, 255, 255, 255],  # 5 construction_vehicle cyan
        [200, 180, 0, 255],  # 6 motorcycle           dark orange
        [255, 0, 0, 255],  # 7 pedestrian           red
        [255, 240, 150, 255],  # 8 traffic_cone         light yellow
        [135, 60, 0, 255],  # 9 trailer              brown
        [160, 32, 240, 255],  # 10 truck                purple
        [255, 0, 255, 255],  # 11 driveable_surface    dark pink
        # [175,   0,  75, 255],  #                   dark red
        [139, 137, 137, 255], # 12 other_flat         灰白色
        [75, 0, 75, 255],  # 13 sidewalk             dard purple
        [150, 240, 80, 255],  # 14 terrain              light green
        [230, 230, 250, 255],  # 15 manmade              white
        [0, 175, 0, 255],  # 16 vegetation           green
        [0, 255, 127, 255],  # 17 ego car              dark cyan
        [255, 99, 71, 255], # 18
        [0, 191, 255, 255], # 19
        [255,255,255,255] # 20
    ]
).astype(np.uint8)

dict = {
        'human.pedestrian.adult': '7',
        'human.pedestrian.child': '7',
        'human.pedestrian.wheelchair': '7',
        'human.pedestrian.stroller': '7',
        'human.pedestrian.personal_mobility': '7',
        'human.pedestrian.police_officer': '7',
        'human.pedestrian.construction_worker': '7',
        'vehicle.bicycle': '2',
        'vehicle.motorcycle': '6',
        'vehicle.car': '4',
        'vehicle.bus.bendy': '3',
        'vehicle.bus.rigid': '3',
        'vehicle.truck': '10',
        'vehicle.emergency.ambulance': '10', # 算到truck里
        'vehicle.emergency.police': '4', # 算到car里
        'vehicle.construction': '5',  # 工程用车，挖掘机啥的
        'vehicle.trailer': '9',
        'movable_object.barrier': '1',
        'movable_object.trafficcone': '8',
        
        'animal': '0', # 算到background里
        'movable_object.pushable_pullable': '0',
        'movable_object.debris': '0',
        'tatic_object.bicycle_rack': '0',
        'static_object.bicycle_rack':'0',
    }

#mlab.options.offscreen = True

voxel_size = 1.0
pc_range = [-50, -50,  -5, 50, 50, 3]

#-----------------------------------------------------------------------------#
# 辅助函数
def EulerAndQuaternionTransform(intput_data): # 四元数与欧拉角互转
    data_len = len(intput_data)
    angle_is_not_rad = True # True:角度值 False:弧度制
    if data_len == 3:
        r,p,y = 0,0,0
        if angle_is_not_rad: # 180 ->pi
            r,p,y = math.radians(intput_data[0]),math.radians(intput_data[1]),math.radians(intput_data[2])
        else:
            r,p,y = intput_data[0],intput_data[1],intput_data[2]
 
        sinp = math.sin(p/2)
        siny = math.sin(y/2)
        sinr = math.sin(r/2)

        cosp = math.cos(p/2)
        cosy = math.cos(y/2)
        cosr = math.cos(r/2)
 
        w = cosr*cosp*cosy + sinr*sinp*siny
        x = sinr*cosp*cosy - cosr*sinp*siny
        y = cosr*sinp*cosy + sinr*cosp*siny
        z = cosr*cosp*siny - sinr*sinp*cosy
        return [w,x,y,z]
 
    if data_len == 4:
        w,x,y,z = intput_data[0],intput_data[1],intput_data[2],intput_data[3]
 
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - z * x))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
 
        if angle_is_not_rad : # pi -> 180
            r,p,y = math.degrees(r),math.degrees(p),math.degrees(y)
        return [r,p,y]

def rotate_points(points, center, rotation_angle): # 将点云中的点围绕指定中心点进行水平旋转
    # rotation_angle为旋转角度（弧度制）
    
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                [0, 0, 1]])
    translated_points = np.asarray(points) - center
    rotated_points = np.dot(rotation_matrix, translated_points.T).T
    rotated_points += center
    return rotated_points

def color_points_in_bbox(points, bbox_center, bbox_size, bbox_rotation, label_name, color):
    # bbox_rotation为旋转角度（弧度制）
    
    rotated_points = rotate_points(points, bbox_center, bbox_rotation)
    bbox_min = bbox_center - bbox_size*0.5
    bbox_max = bbox_center + bbox_size*0.5
    inside_bbox = (np.all(rotated_points >= bbox_min, axis=1) & np.all(rotated_points <= bbox_max, axis=1))
    color[inside_bbox] = label_name
    return color


#-----------------------------------------------------------------------------#
def image_to_video(image_path, media_path, fps=30): # 图像拼接生成视频
    names = os.listdir(image_path) # 获取图片路径下面的所有图片名称
    image_names = []
    for name in names:
        if name[-4:] == '.png':
            image_names.append(name)
    
    image_names.sort(key=lambda n: int(n.split('_')[-1][:-4]))
    # image_names.sort() # 对提取到的图片名称进行排序
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V') # 设置写入格式
    
    image = Image.open(image_path + image_names[0]) # 读取第一个图片获取大小尺寸，因为需要转换成视频的图片大小尺寸是一样的
    media_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size) # 初始化媒体写入对象
    for image_name in image_names: # 遍历图片，将每张图片加入视频当中
        im = cv2.imread(os.path.join(image_path, image_name))
        media_writer.write(im)
        print(image_name, '合并完成！')
    media_writer.release() # 释放媒体写入对象
    print('无声视频写入完成！')


def color(pointcloud,lidar_labels,label_names):
    color = np.ones_like(pointcloud[:, 0]) * [0] # 初始颜色为黑色
    for i in range(len(lidar_labels)):
        lidar_label = lidar_labels[i] # lidar_label前8位是bbox8个顶点，9位是中心点，10位是长宽高，11位是四元数表示的方向角
        label_name = label_names[i]
        points = pointcloud[:, :3]
        bbox_center,bbox_size,bbox_rotation = np.array(lidar_label[8]),np.array(lidar_label[9]),EulerAndQuaternionTransform(lidar_label[10])
        bbox_rotation = (90-bbox_rotation[-1])*3.14/180 # 角度转弧度
        
        # bbox_center = np.array([-bbox_center[0],bbox_center[0],bbox_center[2]])
        # bbox_size = bbox_size*2
        # bbox_rotation = 90-bbox_rotation
        
        # if i == 0:
        #     bbox_center,bbox_size,label_name = np.array([0,0,0]),np.array([50,50,20]),7
            
        
        
        color = color_points_in_bbox(points, bbox_center, bbox_size, bbox_rotation, label_name, color)
        
        # break
    return color

def visual_lidar(visual_path, lidar_labels, label_names, img_name=None, If_show=True):
    pointcloud = np.fromfile(visual_path, dtype=np.float32, count=-1).reshape([-1, 5])
    
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    s = pointcloud[:, 3]  # 强度
    t = pointcloud[:, 4]  # 时间戳

    c = color(pointcloud,lidar_labels,label_names) # 颜色类别
    
    mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1)) # 设置画布大小和底色
    plt_plot_fov = mlab.points3d(
        x, 
        y, 
        z,
        c,  # Values used for Color
        # colormap='spectral',
        colormap="viridis",
        scale_factor=1, # 缩放系数
        mode="point",
        opacity=1.0, # 不透明度
        vmin=0, # 着色的最大最小值
        vmax=20,
    )
    
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    
    # azimuth、elevation为俯仰角、distance为相机距离、focalpoint为观察点
    mlab.view(figure=None, azimuth=-37.5, elevation=30, distance=250, focalpoint=(0, 0, 0), reset_roll=True)
    # mlab.view(figure=None, azimuth=-37.5, elevation=30, distance=400, focalpoint=(50,50,0), reset_roll=True) # all视角

    
    if img_name:
        mlab.savefig(img_name)
    if If_show:
        mlab.show()



def visual_occ(visual_path, img_name=None, If_show=True):
    fov_voxels = np.load(visual_path, allow_pickle=True)

    fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]

    # figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # pdb.set_trace()
    
    # plt_plot_fov = mlab.points3d(
    #     fov_voxels[:, 0],
    #     fov_voxels[:, 1],
    #     fov_voxels[:, 2],
    #     fov_voxels[:, 3],
    #     colormap="viridis",
    #     scale_factor=voxel_size - 0.05*voxel_size,
    #     mode="cube",
    #     opacity=1.0,
    #     vmin=0,
    #     vmax=20,
    # )

    # plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    # plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
#-----------------------------------------------------------------------------#    
    # 添加自车栅格
    ego_points = []
    for ego_x in range(-3,3,1):
        for ego_y in range(-5,5,1):
            for ego_z in range(-3,3,1):
                point = [ego_x+50,ego_y+50,ego_z+3,ego_z+ego_x+6] # occ车身与原点的水平偏移为(50,50)
                ego_points.append(point)
    
    ego_points = np.array(ego_points)
    plt_plot_fov1 = mlab.points3d(
        ego_points[:,0],
        ego_points[:,1],
        ego_points[:,2],
        ego_points[:,3],
        colormap="viridis",
        scale_factor=voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=20,
    )
    plt_plot_fov1.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov1.module_manager.scalar_lut_manager.lut.table = colors
#-----------------------------------------------------------------------------#    
    # azimuth、elevation为俯仰角、distance为相机距离、focalpoint为观察点
    mlab.view(figure=None, azimuth=-90, elevation=0, distance=400, focalpoint=(50, 50, 0), reset_roll=True) # top视角
    # mlab.view(figure=None, azimuth=-37.5, elevation=30, distance=400, focalpoint=(50,50,0), reset_roll=True) # all视角
    # mlab.view(figure=None, azimuth=0, elevation=30, distance=250, focalpoint=(50,50,0), reset_roll=True) # main视角
    # mlab.view(figure=None, azimuth=-90, elevation=80, distance=20, focalpoint=(50, 50, 3), reset_roll=True) # float视角
        
    if img_name:
        mlab.savefig(img_name)
    if If_show:
        mlab.show()

#-----------------------------------------------------------------------------#
def main():
    # # scene_token = "73030fb67d3c46cfb5e590168088ae39" # 修改可视化场景的 scene token
    # scene_token = "c3e0e9f6ee8d4170a3d22a6179f1ca3a"
    # dataset_version = 'v1.0-trainval' # 修改数据集版本与路径
    # dataset_root = '/home/xingchen/Study/pytorch/SurroundOcc/data/nuscenes'
    dataset_occ_root = '/home/xingchen/Study/pytorch/SurroundOcc/data/nuscenes_occ/samples'
    
    scene_token = "fcbccedd61424f1b85dcbf8f897f9754" # 修改可视化场景的 scene token
    # scene_token = "cc8c0bf57f984915a77078b10eb33198"
    dataset_version = 'v1.0-mini' # 修改数据集版本与路径
    dataset_root = '/home/xingchen/Study/pytorch/SurroundOcc/data/nuscenes/others/v1.0-mini'
    
    fps = 2 # 设置保存视频的每秒帧数
    If_save = True
    If_show = True
    # If_show = False
#-----------------------------------------------------------------------------#
    from nuscenes import NuScenes # 使用任意场景
    nusc = NuScenes(version=dataset_version, dataroot=dataset_root, verbose=True)
    
    scene = nusc.get('scene',scene_token)
    scene_name = scene['name']
    save_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/'+scene_name # 可视化结果保存文件夹位置
    
    sample_tokens = nusc.field2token('sample', 'scene_token', scene_token) # 获取当前 scene 的所有关键帧samples 的 token
    
    lidar_paths, occ_paths, all_lidar_labels, all_label_names = [], [], [], []
    for i in range(len(sample_tokens)):
        sample_token = sample_tokens[i]
        sample_data = nusc.get('sample', sample_token)
        lidar_token = sample_data['data']['LIDAR_TOP'] # 获取当前 sample 的 lidar token
        lidar_data = nusc.get('sample_data',lidar_token)
        
        lidar_filename = lidar_data['filename'] # 获取当前 lidar 的 相对路径
        lidar_path = dataset_root+'/'+lidar_filename # 改成绝对路径
        lidar_paths.append(lidar_path)
        
        occ_path = dataset_occ_root+'/'+lidar_filename[18:]+'.npy' # 获取当前 occ 的绝对路径
        occ_paths.append(occ_path)
        
        # # 另存环绕图像与点云
        # for camera in ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT','LIDAR_TOP']:
        #     camera_token = sample_data['data'][camera]
        #     camera_data = nusc.get('sample_data',camera_token)
        #     camera_path = camera_data['filename']
        #     os.makedirs(save_dir+'/'+camera, exist_ok=True)
        #     if camera != 'LIDAR_TOP':
        #         shutil.copy(dataset_root+'/'+camera_path, save_dir+'/'+camera+'/'+scene_name+'_'+str(i)+'.jpg')
        #     else:
        #         shutil.copy(dataset_root+'/'+camera_path, save_dir+'/'+camera+'/'+scene_name+'_'+str(i)+'.bin')
        # # 另存占据标签
        # os.makedirs(save_dir+'/OCC', exist_ok=True)
        # shutil.copy(occ_path, save_dir+'/OCC/'+scene_name+'_'+str(i)+'.npy')
        # print('cameras and lidar and occ are saved !')
            
        
        lidar_anns_tokens = sample_data['anns'] # 当前 sample 的 anns tokens # ef63a697930c4b20a6b9791f423351da
        ego_pose_token = lidar_data['ego_pose_token']
        pose_rec = nusc.get('ego_pose', ego_pose_token) # 车身Ego位置矩阵
        
        lidar_calibrated_token = lidar_data['calibrated_sensor_token']
        lidar_rec = nusc.get('calibrated_sensor', lidar_calibrated_token) # 雷达相对车身位置矩阵
        
        lidar_labels, label_names = [], []
        
        for lidar_anns_token in lidar_anns_tokens:
            # lidar_anns = nusc.get('sample_annotation', lidar_anns_token) 
            
            box = nusc.get_box(lidar_anns_token) # 世界坐标系下
            # 把 世界坐标系 下坐标变成 Ego自身坐标系 下坐标
            box.translate(-np.array(pose_rec['translation']))
            box.rotate(Quaternion(pose_rec['rotation']).inverse)
            
            # 把 Ego坐标系 下坐标变成 lidar坐标系 下坐标
            box.translate(-np.array(lidar_rec['translation']))
            box.rotate(Quaternion(lidar_rec['rotation']).inverse)
            
            corners_3d = box.corners() # 获取ego坐标系下bbox的8个顶点
            name = int(dict[box.name])
            label_names.append(name)
            
            points = [] # 调整8个顶点格式
            for i in range(8):
                point = [corners_3d[0][i],corners_3d[1][i],corners_3d[2][i]]
                points.append(point)
            
            points.append(list(box.center)) # bbox中心点
            points.append(list(box.wlh)) # bbox长宽高
            points.append(list(box.orientation)) # bbox方向 四元数表示
            
            lidar_label = points
            lidar_labels.append(lidar_label) # 当前 sample 的所有labels
        all_lidar_labels.append(lidar_labels) # 当前 scene 的所有labels
        all_label_names.append(label_names)
    
#-----------------------------------------------------------------------------#    
    save_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/scene-0002-easy' # 使用固定场景
    scene_name = 'scene-0002'
    
    occ_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/'+ scene_name +'/OCC'
    occ_paths = [occ_dir+'/'+path for path in os.listdir(occ_dir)]
    occ_paths.sort(key=lambda n: int(n.split('_')[-1][:-4]))
    
    # lidar_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/'+ scene_name +'/'+'LIDAR_TOP'
    # lidar_paths = [lidar_dir+'/'+path for path in os.listdir(lidar_dir)]
    # lidar_paths.sort(key=lambda n: int(n.split('_')[-1][:-4]))
    
    # all_cameras = [] # 当前 scene 的所有环绕图像
    # for i in range(len(lidar_paths)): 
    #     cameras = [] # 当前 sample 的6张环绕图像
    #     for camera in ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']:
    #         camera_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/'+ scene_name +'/'+camera
    #         camera_paths = [camera_dir+'/'+path for path in os.listdir(camera_dir)]
    #         camera_paths.sort(key=lambda n: int(n.split('_')[-1][:-4]))
    #         cameras.append(camera_paths[i])
    #     all_cameras.append(cameras)

#-----------------------------------------------------------------------------#
    # for i in range(len(occ_paths)): # 可视化occ
    #     occ_path = occ_paths[i]
    #     if If_save:
    #         os.makedirs(save_dir+'/'+'occ', exist_ok=True)
    #         img_name = save_dir+'/'+'occ'+'/'+scene_name+'_'+'occ'+'_'+str(i)+'.png'
    #     else:
    #         img_name = None
    #     print(occ_path)
    #     visual_occ(occ_path, img_name, If_show)
    #     # if i>=0:
    #     #     break    
    
    # if If_save and True:
    #     media_path = save_dir+'/'+'occ'+'/'+scene_name+'_'+'occ'+'.mp4'
    #     image_path = save_dir+'/'+'occ'+'/'
    #     image_to_video(image_path, media_path, fps)
#-----------------------------------------------------------------------------#   
    for i in range(len(lidar_paths)): # 可视化lidar
        lidar_path = lidar_paths[i]
        
        lidar_labels = all_lidar_labels[i]
        label_names = all_label_names[i]
        
        if If_save:
            os.makedirs(save_dir+'/'+'lidar', exist_ok=True)
            img_name = save_dir+'/'+'lidar'+'/'+scene_name+'_'+'lidar'+'_'+str(i)+'.png'
        else:
            img_name = None
        print(lidar_path)
        visual_lidar(lidar_path, lidar_labels, label_names, img_name, If_show)
        if i>=0:
            break
    
    # if If_save and True:
    #     media_path = save_dir+'/'+'lidar'+'/'+scene_name+'_'+'lidar'+'.mp4'
    #     image_path = save_dir+'/'+'lidar'+'/'
    #     image_to_video(image_path, media_path, fps)
#-----------------------------------------------------------------------------# 


        
    
    
    
    


if __name__ == "__main__":
    main()
    
    
    
