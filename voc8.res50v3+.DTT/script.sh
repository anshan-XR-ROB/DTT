#!/usr/bin/env bash
nvidia-smi

# export os.environ['CUDA_VISIBLE_DEVICES'] = '1'
export volna=""
export NGPUS=4
export OUTPUT_PATH=""
export snapshot_dir=$OUTPUT_PATH/

export batch_size=2
export learning_rate=0.0025
# export learning_rate=0.001

export snapshot_iter=1
#python train_single.py --nproc_per_node=$NGPUS
python tri_training.py
#python -m torch.distributed.launch --nproc_per_node=$NGPUS train_single.py 
#export TARGET_DEVICE=$[$NGPUS-1]
#python eval.py -e 39 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results
#python eval.py -e 30-60 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results

# following is the command for debug
# export NGPUS=1
# export batch_size=2
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1
