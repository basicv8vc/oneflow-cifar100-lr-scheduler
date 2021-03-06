export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=2
PORT=12345
MODEL_ARCH="resnet34"

python3 -m oneflow.distributed.launch --nproc_per_node $GPU_NUMS --master_port $PORT  \
        main.py --model_arch $MODEL_ARCH --checkpoint ./output/2022-05-30-23:45:48/resnet34.epoch-60.flow --start_epoch 61