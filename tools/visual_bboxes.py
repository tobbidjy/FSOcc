# 可视化点云与图像上的标签bbox, 可视化点云分割与占据预测的codes也保留

import shutil
import os, sys
import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
import torch
from PIL import Image
from pyquaternion.quaternion import Quaternion
import math
from nuscenes.utils.geometry_utils import view_points
from typing import List, Tuple, Union
from shapely.geometry import MultiPoint, box


#-----------------------------------------------------------------------------#
colors = np.array(
    [ # R G B
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
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255], # 12
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

def post_process_coords(corner_coords: List, # 把3Dbbox坐标转换成2Dbbox
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        try:
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])
        except:
            min_x, min_y, max_x, max_y = 0,0,1600,900

        # intersection_coords = img_intersection.bounds
        # min_x, min_y, max_x, max_y = intersection_coords
        return min_x, min_y, max_x, max_y
    else:
        return None
        
def draw_3d_bbox(img,corner_coords,label_color): # 可以直接画框的，这里画线是懒得改了
    points = [(int(corner_coords[0][0]), int(corner_coords[0][1])), 
              (int(corner_coords[1][0]), int(corner_coords[1][1])), 
              (int(corner_coords[2][0]), int(corner_coords[2][1])), 
              (int(corner_coords[3][0]), int(corner_coords[3][1])), 
              (int(corner_coords[4][0]), int(corner_coords[4][1])), 
              (int(corner_coords[5][0]), int(corner_coords[5][1])), 
              (int(corner_coords[6][0]), int(corner_coords[6][1])), 
              (int(corner_coords[7][0]), int(corner_coords[7][1]))]
    # 将边界框的点连接起来
    cv2.line(img, points[0], points[1], label_color, 1)
    cv2.line(img, points[1], points[2], label_color, 1)
    cv2.line(img, points[2], points[3], label_color, 1)
    cv2.line(img, points[3], points[0], label_color, 1)
    cv2.line(img, points[4], points[5], label_color, 1)
    cv2.line(img, points[5], points[6], label_color, 1)
    cv2.line(img, points[6], points[7], label_color, 1)
    cv2.line(img, points[7], points[4], label_color, 1)
    cv2.line(img, points[0], points[4], label_color, 1)
    cv2.line(img, points[5], points[1], label_color, 1)
    cv2.line(img, points[6], points[2], label_color, 1)
    cv2.line(img, points[7], points[3], label_color, 1)
    return img

def draw_2d_bbox(img,final_coords,label_color):
    min_x, min_y, max_x, max_y = list(final_coords)
    
    points = [(int(min_x),int(max_y)),
              (int(max_x),int(max_y)),
              (int(max_x),int(min_y)),
              (int(min_x),int(min_y))]
    # 将边界框的点连接起来
    cv2.line(img, points[0], points[1], label_color, 1)
    cv2.line(img, points[1], points[2], label_color, 1)
    cv2.line(img, points[2], points[3], label_color, 1)
    cv2.line(img, points[3], points[0], label_color, 1)
    return img

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

def cal_color(pointcloud,lidar_labels,label_names): # 确定点云语义分割的各项颜色
    color = np.ones_like(pointcloud[:, 0]) * [0] # 初始颜色为白色
    for i in range(len(lidar_labels)):
        lidar_label = lidar_labels[i] # lidar_label前8位是bbox8个顶点，9位是中心点，10位是长宽高，11位是四元数表示的方向角
        label_name = label_names[i]
        points = pointcloud[:, :3]
        bbox_center,bbox_size,bbox_rotation = np.array(lidar_label[8]),np.array(lidar_label[9]),EulerAndQuaternionTransform(lidar_label[10])
        bbox_rotation = (90-bbox_rotation[-1])*3.14/180 # 角度转弧度
        color = color_points_in_bbox(points, bbox_center, bbox_size, bbox_rotation, label_name, color)
        
        # break
    return color

#-----------------------------------------------------------------------------#

def plot3Dbox_camera(image_path, all_corner_coords, all_final_coords, label_names, img_name_3D, img_name_2D): # 可视化3Dbox
    img = cv2.imread(image_path)
    for i in range(len(all_corner_coords)):
        corner_coords = all_corner_coords[i]
        label_name = label_names[i]
        label_color = colors[label_name][:3]
        label_color = (int(label_color[2]),int(label_color[1]),int(label_color[0])) # 不加int()有bug
        img = draw_3d_bbox(img,corner_coords,label_color)
    if img_name_3D:
        cv2.imwrite(img_name_3D,img)
    
    # cv2.imshow("Bounding Box1", img) # 可视化有问题
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    # 可视化2Dbox
    img = cv2.imread(image_path)
    for i in range(len(all_final_coords)):
        final_coords = all_final_coords[i]
        label_name = label_names[i]
        label_color = colors[label_name][:3]
        label_color = (int(label_color[2]),int(label_color[1]),int(label_color[0]))  # B G R
               
        img = draw_2d_bbox(img,final_coords,label_color)
        
        # break # 只可视化一个bbox
    if img_name_2D:
        cv2.imwrite(img_name_2D,img)
    # cv2.imshow("Bounding Box2", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows() 

def plot3Dbox_lidar(lidar_labels, label_names, If_show, img_name):
    for i in range(len(label_names)):
        lidar_label = lidar_labels[i][:8]
        label_name = label_names[i]
        label_color = colors[label_name][:3]/255
        label_color = (label_color[0],label_color[1],label_color[2])

        corner = np.array(lidar_label).T
        # corner = np.array([[0,100,100,0,0,100,100,0],[0,0,100,100,0,0,100,100],[0,0,0,0,100,100,100,100]])
        idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]) # 绘制bbox时顶点两两连接的顺序
        x = corner[0, idx]
        y = corner[1, idx]
        z = corner[2, idx]
        
        mlab.plot3d(
            x, 
            y, 
            z, 
            color = label_color, 
            colormap='spectral', 
            representation='wireframe',
            line_width=1,
            opacity=1.0, # 不透明度
            vmin=0, # 着色的最大最小值
            vmax=20,
        )
        # break
    mlab.view(figure=None, azimuth=-37.5, elevation=30, distance=250, focalpoint=(0, 0, 0), reset_roll=True)

    if img_name:
        mlab.savefig(img_name)
    if If_show:
        mlab.show()

def visual_lidar(visual_path, lidar_labels, label_names, img_name=None, If_show=True, If_bbox=False):
    pointcloud = np.fromfile(visual_path, dtype=np.float32, count=-1).reshape([-1, 5])
    
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    s = pointcloud[:, 3]  # 强度
    t = pointcloud[:, 4]  # 时间戳

    c = cal_color(pointcloud,lidar_labels,label_names) # 颜色类别
    
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

    if If_bbox: # 展示bboxes
        plot3Dbox_lidar(lidar_labels, label_names, If_show, img_name) 
    else: # 不展示bboxes    
        if If_show:
            mlab.show() 
        if img_name:
            mlab.savefig(img_name)

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
    # scene_token = "73030fb67d3c46cfb5e590168088ae39" # 修改可视化场景的 scene token
    scene_token = "c3e0e9f6ee8d4170a3d22a6179f1ca3a"
    dataset_version = 'v1.0-trainval' # 修改数据集版本与路径
    dataset_root = '/home/xingchen/Study/pytorch/SurroundOcc/data/nuscenes'
    dataset_occ_root = '/home/xingchen/Study/pytorch/SurroundOcc/data/nuscenes_occ/samples'
    
    # scene_token = "fcbccedd61424f1b85dcbf8f897f9754" # 修改可视化场景的 scene token
    # # scene_token = "cc8c0bf57f984915a77078b10eb33198"
    # dataset_version = 'v1.0-mini' # 修改数据集版本与路径
    # dataset_root = '/home/xingchen/Study/pytorch/SurroundOcc/data/nuscenes/others/v1.0-mini'
    
    fps = 2 # 设置保存视频的每秒帧数
    If_save = True
    # If_show = True
    If_show = False
    If_bbox = True
#-----------------------------------------------------------------------------#
    from nuscenes import NuScenes # 使用任意场景
    nusc = NuScenes(version=dataset_version, dataroot=dataset_root, verbose=True)
    
    scene = nusc.get('scene',scene_token)
    scene_name = scene['name']
    save_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/'+scene_name # 可视化结果保存文件夹位置
    
    sample_tokens = nusc.field2token('sample', 'scene_token', scene_token) # 获取当前 scene 的所有关键帧samples 的 token
    
    lidar_paths, occ_paths, all_image_tokens, all_lidar_labels, all_label_names = [], [], [], [], []
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
        
        image_tokens= []
        for camera in ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']:
            camera_token = sample_data['data'][camera]
            camera_data = nusc.get('sample_data',camera_token)
            camera_path = camera_data['filename']
            image_tokens.append(camera_token)
        all_image_tokens.append(image_tokens)
            
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
    # save_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/scene-0002-easy' # 使用固定场景
    # scene_name = 'scene-0002'
    
    # occ_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/'+ scene_name +'/OCC'
    # occ_paths = [occ_dir+'/'+path for path in os.listdir(occ_dir)]
    # occ_paths.sort(key=lambda n: int(n.split('_')[-1][:-4]))
    
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
            if If_bbox: # 可视化3Dbbox
                os.makedirs(save_dir+'/'+'lidar_bbox', exist_ok=True)
                img_name = save_dir+'/'+'lidar_bbox'+'/'+scene_name+'_'+'lidar_bbox'+'_'+str(i)+'.png'
            else:
                os.makedirs(save_dir+'/'+'lidar', exist_ok=True)
                img_name = save_dir+'/'+'lidar'+'/'+scene_name+'_'+'lidar'+'_'+str(i)+'.png'
        else:
            img_name = None
        print(lidar_path)
        visual_lidar(lidar_path, lidar_labels, label_names, img_name, If_show, If_bbox)
        # if i>=0:
        #     break
    
    # if If_save:
    #     media_path = save_dir+'/'+'lidar'+'/'+scene_name+'_'+'lidar'+'.mp4'
    #     image_path = save_dir+'/'+'lidar'+'/'
    #     image_to_video(image_path, media_path, fps)
#-----------------------------------------------------------------------------# 
    pass

    for i in range(len(all_image_tokens)): # 可视化camera
        image_tokens = all_image_tokens[i] # 当前 sample 的6张环绕图像的 token
        
        sample_token = sample_tokens[i] # 获取当前sample的lidar标签
        sample_data = nusc.get('sample', sample_token)
        lidar_anns_tokens = sample_data['anns'] # 当前 sample 的 anns tokens # ef63a697930c4b20a6b9791f423351da
        
        CAM_token = sample_data['data']['CAM_FRONT'] # 获取当前 sample 的 CAM token
        CAM_data = nusc.get('sample_data',CAM_token)
        ego_pose_token = CAM_data['ego_pose_token']
        pose_rec = nusc.get('ego_pose', ego_pose_token) # 车身Ego位置矩阵
        
        for h in range(len(image_tokens)): # 取6张环绕图像中的1张
            image_token = image_tokens[h]
            image_data = nusc.get('sample_data',image_token)
            image_filename = image_data['filename'] # 获取当前 image 的 相对路径
            image_path = dataset_root+'/'+image_filename # 改成绝对路径
        
            camera_calibrated_token = image_data['calibrated_sensor_token']
            camera_rec = nusc.get('calibrated_sensor', camera_calibrated_token) # 车身Ego相对相机位置矩阵 相机外参

            camera_intrinsic = np.array(camera_rec['camera_intrinsic']) # 相机到像素位置矩阵 相机内参
            all_corner_coords, all_final_coords, label_names = [], [], []
            for j in range(len(lidar_anns_tokens)): # 获取所有bbox, 相机的标签与雷达标签都是从 世界坐标系 下的标签转换来的
                anns_token = lidar_anns_tokens[j]
                anns = nusc.get('sample_annotation', anns_token)
                if anns['visibility_token'] not in ['2', '3', '4']:
                    continue

                box = nusc.get_box(anns_token) # 世界坐标系下
                name = int(dict[box.name]) # 获取标签类别
                
                # 把 世界坐标系 下坐标变成 Ego自身坐标系 下坐标
                box.translate(-np.array(pose_rec['translation']))
                box.rotate(Quaternion(pose_rec['rotation']).inverse)
                
                # 把 Ego坐标系 下坐标变成 相机坐标系 下坐标
                box.translate(-np.array(camera_rec['translation']))
                box.rotate(Quaternion(camera_rec['rotation']).inverse)
                
                # 过滤掉不在校准传感器前方的bbox
                corners_3d = box.corners()
                in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                corners_3d = corners_3d[:, in_front] # 获取在当前CAM前方的角点

                # 相机坐标系 下坐标变成 像素坐标系 下坐标
                corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist() # 3Dbbox
                if corner_coords is None or len(corner_coords)<8: # 过滤一些错误
                    continue
                    
                final_coords = post_process_coords(corner_coords) # 2Dbbox
                if final_coords is None or len(final_coords)<4:# 过滤一些错误
                    continue

                all_corner_coords.append(corner_coords) # 保存当前图像所有需要绘制的3Dbbox
                all_final_coords.append(final_coords) # 保存当前图像所有需要绘制的2Dbbox
                label_names.append(name) # 保存当前图像所有需要绘制的bbox的类别
            if If_save:
                camera_kind = image_path.split('__')[-2]
                os.makedirs(save_dir+'/3Dbbox_'+camera_kind, exist_ok=True)
                os.makedirs(save_dir+'/2Dbbox_'+camera_kind, exist_ok=True)
                img_name_3D = save_dir+'/3Dbbox_'+camera_kind+'/'+scene_name+'_'+str(i)+'.png'
                img_name_2D = save_dir+'/2Dbbox_'+camera_kind+'/'+scene_name+'_'+str(i)+'.png'
            else:
                img_name_3D, img_name_2D = None, None

            if If_show: # 绘制所有投影的bbox
                plot3Dbox_camera(image_path, all_corner_coords, all_final_coords, label_names, img_name_3D, img_name_2D)
                print(img_name_3D,'and',img_name_2D, 'are saved!')
        # break # 只可视化一个sample


if __name__ == "__main__":
    main()
    
    
    
