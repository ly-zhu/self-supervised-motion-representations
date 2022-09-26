#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "

OPTS+="--list_train data/MUSIC21_train.csv "
OPTS+="--list_val data/MUSIC21_val.csv "

# Models
OPTS+="--arch_sound resnet182d_A "
OPTS+="--arch_frame resnet183d_V "

# logscale in frequency
OPTS+="--num_mix 1 "
OPTS+="--log_freq 1 "

OPTS+="--dataset MUSIC21 "

# frames-related
OPTS+="--num_frames 48 "
OPTS+="--stride_frames 1 "
OPTS+="--frameRate 8 "

OPTS+="--margin_dur 8 "
OPTS+="--vid_dur 48 " 
OPTS+="--shift_dur 8 " 
OPTS+="--non_inter_dur 8 " 

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "
OPTS+="--stft_frame 1022 "
OPTS+="--stft_hop 256 "

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--workers 12 "
OPTS+="--batch_size_per_gpu 10 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-4 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 40 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

OPTS+="--ckpt ./ckpt_ame_MUSIC21_scratch "
OPTS+="--dup_trainset 100 "

# data loading related
# diff_same: 0 # load diff and same equal randomly
# diff: 1
# same: 2
OPTS+="--dataFlag 2 "
# 1: random shift
# 0: no shift
OPTS+="--audShiftFlag 1 "
OPTS+="--shiftRegressionFlag 0 "

CUDA_VISIBLE_DEVICES="0" python -u main_AME.py $OPTS
