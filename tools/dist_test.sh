# 评估三维语义占据预测  3D semantic occupancy prediction
# bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc.py ckpts/surroundocc.pth 1
#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2 # weight
GPUS=$3 # 1
PORT=${PORT:-29282}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --eval bbox

