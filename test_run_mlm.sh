#!/bin/sh
#COBALT -n 1
#COBALT -q full-node
#COBALT -A SolarWindowsADSP
#COBALT -t 30
#COBALT -M shu.huang-2@outlook.com
#COBALT --attrs pubnet

PYTHON=/lus/theta-fs0/projects/SolarWindowsADSP/conda/envs/bert-pytorch/bin/python
PYTHON_PROGRAM=run_mlm.py
INPUT_FILE=/projects/SolarWindowsADSP/shu/papertext/train/train1.txt
VALIDATION_FILE=/projects/SolarWindowsADSP/shu/papertext/validation/validation1.txt
MODEL_DIR=bert-base-cased
OUTPUT_DIR=/projects/SolarWindowsADSP/shu/models/batteryvocab-cased-mlm-new/
CACHE_DIR=/lus/theta-fs0/projects/SolarWindowsADSP/shu/models/cache/

nvidia-smi
unset LD_PRELOAD
${PYTHON} -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 ${PYTHON_PROGRAM} --model_name_or_path ${MODEL_DIR} --train_file ${INPUT_FILE} --validation_file ${VALIDATION_FILE} --output_dir  ${OUTPUT_DIR} --max_seq_length 512 --cache_dir ${CACHE_DIR} --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 1e-4 --weight_decay 0.01 --num_train_epochs 40 --max_steps 1000000 --save_steps 10000 --fp16 True
