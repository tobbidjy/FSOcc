# 将inference后的整个场景各帧的pred.npy文件保存成image

import os, sys
import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
import torch


colors = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

def get_image_and_save(visual_path,image_path):
    voxel_size = 1
    pc_range = [-50, -50,  -5, 50, 50, 3]
    
    fov_voxels = np.load(visual_path)
    
    fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]

    #figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # pdb.set_trace()
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05*voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )


    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    
    # mlab.options.offscreen = True # 离屏渲染

    # azimuth、elevation为俯仰角、distance为相机距离、focalpoint为观察点
    mlab.view(figure=None, azimuth=-37.5, elevation=30, distance=250, focalpoint=(-50, -50, 0), reset_roll=True)
    # mlab.view(figure=None, azimuth=-37.5, elevation=30, distance=400, focalpoint=(50,50,0), reset_roll=True) # all视角
    mlab.savefig(image_path)
    # mlab.show()
    print(image_path,"is saved")

def image_to_video(image_path, media_path, fps=30): # 图像拼接生成视频
    names = os.listdir(image_path) # 获取图片路径下面的所有图片名称
    image_names = []
    for name in names:
        if name[-4:] == '.png':
            image_names.append(name)
    
    image_names.sort(key=lambda n: int(n.split('_')[-1][:-12]))
    # image_names.sort() # 对提取到的图片名称进行排序
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V') # 设置写入格式
    
    from PIL import Image
    image = Image.open(image_path + image_names[0]) # 读取第一个图片获取大小尺寸，因为需要转换成视频的图片大小尺寸是一样的
    media_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size) # 初始化媒体写入对象
    for image_name in image_names: # 遍历图片，将每张图片加入视频当中
        im = cv2.imread(os.path.join(image_path, image_name))
        media_writer.write(im)
        print(image_name, '合并完成！')
    media_writer.release() # 释放媒体写入对象
    print('无声视频写入完成！')


def main():
    # scene_name = 'scene-0000'
    scene_name = 'scene-0655'
    # scene_name = 'scene-1077'
    # scene_name = 'scene-1094'
    
    
    
    
    visual_dir = '/home/ubuntu/code/xuzeyuan/SurroundOcc/visual_dir/'+scene_name
    visual_paths = [visual_dir+'/origin/'+path+'/pred.npy' for path in os.listdir(visual_dir+'/origin')]
    save_image_dir = visual_dir+'/occ_images/'
    image_paths = [save_image_dir+path+'.png' for path in os.listdir(visual_dir+'/origin')]
    
    save_image = True # 是否生成图像
    save_video = False # 是否生成视频
    save_video_path = visual_dir+'/'+scene_name+'.mp4'

    if save_image:
        for i in range(len(visual_paths)):
            visual_path = visual_paths[i]
            image_path = image_paths[i] 
            get_image_and_save(visual_path,image_path)
            # break
        
    
    if save_video:
        image_to_video(save_image_dir, save_video_path, fps=2)


if __name__ == "__main__":
    main()



