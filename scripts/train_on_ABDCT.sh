#!/bin/bash
# train a model to segment abdominal MRI (T2 fold of CHAOS challenge)
GPUID1=1
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='ABDOMEN_CT'
NWORKER=16
RUNS=1
ALL_EV=(0) # 5-fold cross validation (0, 1, 2, 3, 4)
# TEST_LABEL=[2,3]
EXCLUDE_LABEL=[2,3] # use class 1, 6 as training classes
TEST=1236   # flag for folder name, '1234' for not excluding any labels,  '14' for excluding label [2,3], '23' for excluding label [1,4]
USE_GT=True
###### Training configs ######
NSTEP=200000
DECAY=0.98

MAX_ITER=3000 # defines the size of an epoch
SNAPSHOT_INTERVAL=5000 # interval for saving snapshot
SEED=2025

echo ========================================================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
  PREFIX="train_${DATASET}_cv${EVAL_FOLD}"
  echo $PREFIX
  LOGDIR="./exps_train_on_${DATASET}"

  if [ ! -d $LOGDIR ]
  then
    mkdir -p $LOGDIR
  fi

  python3 train.py with \
  mode='train' \
  dataset=$DATASET \
  num_workers=$NWORKER \
  n_steps=$NSTEP \
  eval_fold=$EVAL_FOLD \
  test_label=$TEST_LABEL \
  exclude_label=$EXCLUDE_LABEL \
  use_gt=$USE_GT \
  max_iters_per_load=$MAX_ITER \
  seed=$SEED \
  save_snapshot_every=$SNAPSHOT_INTERVAL \
  lr_step_gamma=$DECAY \
  path.log_dir=$LOGDIR
done
