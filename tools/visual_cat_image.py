# 将occ与lidar可视化的结果以及原环绕图像拼接在一张图片上，并按场景生成视频

# 切换环境VoxelNeXt运行，可能存在cv2的bug，直接运行失败，debug运行成功

from PIL import Image
import cv2
import numpy as np
import os


bar_path = '/home/xingchen/Study/pytorch/SurroundOcc/assets/Occ_bar.png' # 图例

scene_name = 'scene-0002'

line_color = (0,0,0)
line_thickness = 10

save_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/' +scene_name+'/cat'

fps = 2
#-----------------------------------------------------------------------------#
def image_to_video(image_dir_path, media_path, fps=30): # 图像拼接生成视频
    names = os.listdir(image_dir_path) # 获取图片路径下面的所有图片名称
    image_names = []
    for name in names:
        if name[-4:] == '.png':
            image_names.append(name)
    
    image_names.sort(key=lambda n: int(n[:-4]))
    # image_names.sort() # 对提取到的图片名称进行排序
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V') # 设置写入格式
    image_path = image_dir_path +'/'+ image_names[0]
    image = Image.open(image_path) # 读取第一个图片获取大小尺寸，因为需要转换成视频的图片大小尺寸是一样的
    media_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size) # 初始化媒体写入对象
    for image_name in image_names: # 遍历图片，将每张图片加入视频当中
        im = cv2.imread(os.path.join(image_dir_path, image_name))
        media_writer.write(im)
        print(image_name, '合并完成！')
    media_writer.release() # 释放媒体写入对象
    print('无声视频写入完成！')
#-----------------------------------------------------------------------------#

lidar_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/'+ scene_name +'/'+'LIDAR_TOP'
lidar_paths = [lidar_dir+'/'+path for path in os.listdir(lidar_dir)]
lidar_paths.sort(key=lambda n: int(n.split('_')[-1][:-4]))

all_cameras = [] # 当前 scene 的所有环绕图像+lidar可视化图像
for i in range(len(lidar_paths)): 
    cameras = [] # 当前 sample 的6张环绕图像+lidar可视化图像
    for camera in ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT','lidar_bbox']:
        camera_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/'+ scene_name +'/'+camera
        camera_paths = [camera_dir+'/'+path for path in os.listdir(camera_dir)]
        camera_paths.sort(key=lambda n: int(n.split('_')[-1][:-4]))
        cameras.append(camera_paths[i])
    all_cameras.append(cameras)

all_occ_cameras = [] # 当前 scene 的所有occ图像
for i in range(len(lidar_paths)): 
    occ_cameras = [] # 当前 sample 的4张occ图像
    for camera in ['occ_all','occ_float','occ_main','occ_top']:
        camera_dir = '/home/xingchen/Study/pytorch/SurroundOcc/outputs/'+ scene_name +'/'+camera
        camera_paths = [camera_dir+'/'+path for path in os.listdir(camera_dir)]
        camera_paths.sort(key=lambda n: int(n.split('_')[-1][:-4]))
        occ_cameras.append(camera_paths[i])
    all_occ_cameras.append(occ_cameras)
        
        

camera_position = [[0,450,800,1600],[0,450,1600,2400],[450,900,1600,2400],
                   [450,900,800,1600],[450,900,0,800],[0,450,0,800],
                   [0,572,2400,3389]]

occ_camera_position = [[572,1144,2400,3389],[1716,2288,2400,3389],
                   [900,2288,0,2400],[1144,1716,2400,3389]]



for i in range(len(lidar_paths)): 
    cameras = all_cameras[i] # 'CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT','lidar'
    occ_cameras = all_occ_cameras[i] # 'occ_all','occ_float','occ_main','occ_top'
    
    save_path = save_dir+'/'+str(i)+'.png'
    
    image_size = [450*2+1388+235,2400+989] # [x,y]
    image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8)*255
    
    bar = cv2.imread(bar_path)
    bar = cv2.resize(bar,(3389,235))
    image[2288:2523, 0:3389,:] = bar
    
    
 
    for i in range(len(cameras)):
        camera_path = cameras[i]
        camera = cv2.imread(camera_path)
        camera = cv2.resize(camera,(800,450))  if i != 6 else cv2.resize(camera,(989,572))
        image[camera_position[i][0]:camera_position[i][1], camera_position[i][2]:camera_position[i][3],:] = camera

    for i in range(len(occ_cameras)):
        occ_camera_path = occ_cameras[i]
        occ_camera = cv2.imread(occ_camera_path)
        occ_camera = cv2.resize(occ_camera,(989,572)) if i != 2 else cv2.resize(occ_camera,(2400,1388))
        image[occ_camera_position[i][0]:occ_camera_position[i][1], occ_camera_position[i][2]:occ_camera_position[i][3],:] = occ_camera
        
#-----------------------------------------------------------------------------#
    # 绘制边界线
    cv2.rectangle(image, (0,2288), (3389,2523), color=line_color, thickness=line_thickness)
    for i in range(len(cameras)):
        point1,point2 = (camera_position[i][2],camera_position[i][0]),(camera_position[i][3],camera_position[i][1])
        cv2.rectangle(image, point1, point2, color=line_color, thickness=line_thickness)
    for i in range(len(occ_cameras)):
        point1,point2 = (occ_camera_position[i][2],occ_camera_position[i][0]),(occ_camera_position[i][3],occ_camera_position[i][1])
        cv2.rectangle(image, point1, point2, color=line_color, thickness=line_thickness)
#-----------------------------------------------------------------------------#     
    
    cv2.imwrite(save_path,image)
    print(save_path,'is saved !')
    # break

# 拼接视频
media_path = save_dir+'/'+scene_name+'_'+'cat'+'.mp4'
image_path = save_dir
image_to_video(image_path, media_path, fps)














