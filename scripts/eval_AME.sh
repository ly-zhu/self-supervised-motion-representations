#!/bin/bash

OPTS=""
OPTS+="--mode eval "

OPTS+="--ckpt ./ckpt_triplet_shift_1s_res183dScratchKaiming_waveform_N2_f1_binary_MUSIC21_bs10_train888_test195_dup100_f8fps_11k "
OPTS+="--id MUSIC-1mix-LogFreq-resnet18dilated-unet7-linear-frames48stride1-maxpool-binary-weightedLoss-channels21-epoch100-step20_40_80 "

OPTS+="--list_train data/MUSIC21_train.csv "
OPTS+="--list_val data/MUSIC21_test.csv "

# Models
OPTS+="--arch_sound resnet182d_A "
OPTS+="--arch_frame resnet183d_V "

# logscale in frequency
OPTS+="--num_mix 1 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--dataset MUSIC21 "

OPTS+="--num_frames 48 "
OPTS+="--stride_frames 1 "
OPTS+="--frameRate 8 "

OPTS+="--num_gpus 1 "
OPTS+="--workers 12 "
OPTS+="--batch_size_per_gpu 10 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--workers 12 "
OPTS+="--batch_size_per_gpu 1 "
OPTS+="--num_vis 500 "
OPTS+="--num_val 500 "

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
