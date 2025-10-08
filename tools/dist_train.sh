# bash tools/dist_train.sh projects/configs/surroundocc/surroundocc.py 2 /home/ubuntu/code/xuzeyuan/SurroundOcc/ckpts/log_train1

# bash tools/dist_train.sh projects/configs/surroundocc/surroundocc_r50.py 2 /home/ubuntu/code/xuzeyuan/SurroundOcc/ckpts/log_train1


#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
SAVE_PATH=$3

PORT=${PORT:-28108}
NCCL_DEBUG=INFO

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --work-dir ${SAVE_PATH}  --launcher pytorch ${@:4} --deterministic




# resume from
# bash tools/dist_train.sh projects/configs/surroundocc/surroundocc_r50.py 2 /home/ubuntu/code/xuzeyuan/SurroundOcc/ckpts/log_train1 /home/ubuntu/code/xuzeyuan/SurroundOcc/ckpts/log_train1/epoch_2.pth

# CONFIG=$1
# GPUS=$2
# SAVE_PATH=$3
# checkpoint=$4

# PORT=${PORT:-28108}
# NCCL_DEBUG=INFO

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --work-dir ${SAVE_PATH} --resume-from=$checkpoint  --launcher pytorch ${@:4} --deterministic
