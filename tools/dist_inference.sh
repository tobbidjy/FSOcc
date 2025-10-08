# 预测某一场景，场景需要保存到pkl中
# 默认预测的data/infos_inference.pkl是验证集val内所有场景(共150个)
# bash tools/dist_inference.sh ./projects/configs/surroundocc/surroundocc_inference_r50.py ckpts/r50_24epoch_18.37/epoch_22.pth 1
# bash tools/dist_inference.sh ./projects/configs/surroundocc/surroundocc_inference.py ckpts/surroundocc.pth 1

# 想要预测某一场景，用tools/get_a_scene_pkl.py将某个scene的所有关键帧信息保存成pkl文件
# 然后在surroundocc_inference_r50.py中修改预测pkl文件

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29282}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --is_vis  --eval bbox                                                                                                                                                                                                                                                                    py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --is_vis  --eval bbox
