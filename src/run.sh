#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_model.py --model_name \'basic\' --init_seed 0 --control_name \'lstm_200_200_5_0_0_1\' &
CUDA_VISIBLE_DEVICES="1" python train_model.py --model_name \'basic\' --init_seed 0 --control_name \'lstm_200_200_5_0.1_0_1\' &
CUDA_VISIBLE_DEVICES="2" python train_model.py --model_name \'basic\' --init_seed 0 --control_name \'lstm_200_200_5_0.3_0_1\' &
CUDA_VISIBLE_DEVICES="3" python train_model.py --model_name \'basic\' --init_seed 0 --control_name \'lstm_200_200_5_0.5_0_1\' &
CUDA_VISIBLE_DEVICES="0" python train_model.py --model_name \'basic\' --init_seed 0 --control_name \'lstm_200_200_5_0.7_0_1\' &
CUDA_VISIBLE_DEVICES="1" python train_model.py --model_name \'basic\' --init_seed 0 --control_name \'lstm_200_200_5_0.9_0_1\' &
CUDA_VISIBLE_DEVICES="2" python train_model.py --model_name \'basic\' --init_seed 0 --control_name \'lstm_200_200_5_0.95_0_1\' &
CUDA_VISIBLE_DEVICES="3" python train_model.py --model_name \'basic\' --init_seed 0 --control_name \'lstm_200_200_5_1_0_1\' &
