CPCforboat
===================================  
### use main.py to train and save the encoder.
    python main.py \
    --trainset data_pkl/yourdata_train.pickle --testset data_pkl/yourdata_test.pickle \
    --logging_dir result/cpc/ --log_interval 50 --timestep 10 --masked_frames 10 --n_warmup_steps 1000 --epoch 100
### use boat_class.py to train the classifier.
    python boat_class.py \
    --datadir data_pkl --logging-dir result/cpc/ --log-interval 5 --model-path result/cpc/cpc-2022-07-13_21_29_40-model_best.pth --epoch 100
### use result/plotacc.py to plot acc.img.
![Image](https://github.com/tualgfhite/CPCforboat/blob/master/result/classfier.png)

![Image](https://github.com/tualgfhite/CPCforboat/blob/master/result/pretrain.png)
