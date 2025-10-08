# 从infos_inference.pkl中检索某一场景的所有(关键)帧并保存到pkl文件中
import pickle,os



def get_sample_tokens(scene_token): # 获取当前场景的所有关键帧信息
    from nuscenes import NuScenes # 使用任意场景
    # dataset_version = 'v1.0-trainval'
    # dataset_root = '/home/xingchen/Study/pytorch/SurroundOcc/data/nuscenes'
    dataset_version = 'v1.0-mini' # 修改数据集版本与路径
    dataset_root = '/home/ubuntu/code/xuzeyuan/SurroundOcc/data/nuscenes/others/mini'

    nusc = NuScenes(version=dataset_version, dataroot=dataset_root, verbose=True)
    scene = nusc.get('scene',scene_token)
    
    scene_name = scene['name']
    # save_dir = '/home/xingchen/Study/pytorch/SurroundOcc/visual_dir/'+scene_name # 保存文件夹位置
    sample_tokens = nusc.field2token('sample', 'scene_token', scene_token) # 获取当前 scene 的所有关键帧samples 的 token
    return sample_tokens,scene_name

if __name__ == '__main__':
    # scene_token0 = "fcbccedd61424f1b85dcbf8f897f9754" # scene-0103 # 修改场景的 scene token
    # scene_token = "cc8c0bf57f984915a77078b10eb33198"
    
    # scene_token0 = "e7ef871f77f44331aefdebc24ec034b7" 
    # scene_name = 'scene-0000'
    
    scene_token0 = "bebf5f5b2a674631ab5c88fd1aa9e87a" 
    scene_name = 'scene-0655'
    
    
    # scene_token0 = "d25718445d89453381c659b9c8734939"
    # scene_name = 'scene-1077'
    
    # scene_token0 = "de7d80a1f5fb4c3e82ce8a4f213b450a" 
    # scene_name = 'scene-1094'
    
    
    save_sample_data = {'infos': [], 'metadata': {'version': 'v1.0-trainval'}}
    save_sample_data_infos = []
    # with open("/home/xingchen/Study/pytorch/SurroundOcc/data/nuscenes/nuscenes_infos_val.pkl", 'rb') as f:
    with open("/home/ubuntu/code/xuzeyuan/SurroundOcc/data/nuscenes_infos_train.pkl", 'rb') as f:
    
    # with open("/home/ubuntu/code/xuzeyuan/SurroundOcc/data/infos_inference.pkl", 'rb') as f:
        # 反序列化解析成列表scenes_pkl
        scenes_pkl = pickle.load(f)
        no = 1
        for i in range(len(scenes_pkl['infos'])):
            sample = scenes_pkl['infos'][i]
            if sample['scene_token'] == scene_token0:
                print('get sample',no)
                no += 1
                save_sample_data_infos.append(sample)
    save_sample_data['infos'] = save_sample_data_infos
    
    os.makedirs('/home/ubuntu/code/xuzeyuan/SurroundOcc/visual_dir/'+scene_name, exist_ok=True)        
    with open('/home/ubuntu/code/xuzeyuan/SurroundOcc/visual_dir/'+scene_name+'/'+scene_name+".pkl", 'wb') as f:     # 将数据写入pkl文件
        pickle.dump(save_sample_data, f)

    print('OK!')