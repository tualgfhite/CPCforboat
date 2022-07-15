#!/bin/bash
stage="$1" # parse first argument 

if [ $stage -eq 0 ]; then
    # call main.py; CPC train on LibriSpeech
    CUDA_VISIBLE_DEVICES=`free-gpu` python main.py \
--trainset data_pkl/data_train.pickle --testset data_pkl/data_test.pickle \
--logging_dir result/cpc/ --log_interval 50 --timestep 10 --masked_frames 10 --n_warmup_steps 1000 --epoch 100
fi

if [ $stage -eq 1 ]; then
    # call boat_class.py
    CUDA_VISIBLE_DEVICES=`free-gpu` python spk_class.py \
--datadir data_pkl --logging-dir result/cpc/ --log-interval 5 --model-path result/cpc/cpc-2022-07-13_21_29_40-model_best.pth --epoch 100
fi


