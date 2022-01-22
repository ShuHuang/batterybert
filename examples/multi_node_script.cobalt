#!/bin/bash
#COBALT -n 2
#COBALT -q full-node
#COBALT -A SolarWindowsADSP
#COBALT -t 10
#COBALT --attrs pubnet

# Figure out training environment
if [[ -z "${COBALT_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MASTER_RANK=$(head -n 1 $COBALT_NODEFILE)
    RANKS=$(tr '\n' ' ' < $COBALT_NODEFILE)
    NNODES=$(< $COBALT_NODEFILE wc -l)
fi

# Commands to run prior to the Python script for setting up the environment
PYTHON=/lus/theta-fs0/projects/SolarWindowsADSP/conda/envs/bert-pytorch/bin/python
PYTHON_PROGRAM=run_class.py
TRAIN_ROOT=$TRAIN_ROOT
TEST_ROOT=$TEST_ROOT
SAVE_ROOT=$SAVE_ROOT
CHECKPOINT=$CHECKPOINT

nvidia-smi
unset LD_PRELOAD
PRELOAD="export OMP_NUM_THREADS=8;"

# torchrun launch configuration
LAUNCHER="$PYTHON -m torch.distributed.launch "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=8 "
if [[ "$NNODES" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+=""
fi

# Training script and parameters
CMD="$PYTHON_PROGRAM --train_root ${TRAIN_ROOT} --eval_root ${TEST_ROOT} --output_dir ${SAVE_ROOT} --model_name_or_path ${CHECKPOINT}"

# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        echo "Training Command: "$PRELOAD $LAUNCHER --node_rank $RANK $CMD""
        eval "$PRELOAD $LAUNCHER --node_rank $RANK $CMD" &
    else
        echo "Launching rank $RANK on remote node $NODE"
        echo "Training Command: "$PRELOAD $LAUNCHER --node_rank $RANK $CMD""
        ssh $NODE "cd $PWD; $PRELOAD $LAUNCHER --node_rank $RANK $CMD" &
    fi
    RANK=$((RANK+1))
done

wait
