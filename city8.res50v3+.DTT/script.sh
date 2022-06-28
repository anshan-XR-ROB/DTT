#!/usr/bin/env bash
nvidia-smi

# export os.environ['CUDA_VISIBLE_DEVICES'] = '1'
export volna="/data/home/scv3198/code/DTT/"
export NGPUS=2
export OUTPUT_PATH="/data/home/scv3198/code/DTT/CPS-50-1-16-single"
export snapshot_dir=$OUTPUT_PATH/

export batch_size=8
export learning_rate=0.0025
# export learning_rate=0.001

export snapshot_iter=10
#python train_single.py --nproc_per_node=$NGPUS
#python -u tri_training_4.py
#python -u -m torch.distributed.launch --nproc_per_node=$NGPUS train_single.py 
export TARGET_DEVICE=$[$NGPUS-1]
#python eval.py -e 39 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results
python -u eval.py -e 110-240 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results

#python -u tri_training_4.py
# following is the command for debug
# export NGPUS=1
# export batch_size=2
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1
