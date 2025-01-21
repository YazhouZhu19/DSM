#!/bin/bash
# test a model to segment abdominal/cardiac MRI
GPUID1=1
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='' # Multiple Target Medical Datasets
#DATASET='CMR'
NWORKER=16
RUNS=1
ALL_EV=(0) # 5-fold cross validation (0, 1, 2, 3, 4)
TEST_LABEL=[2,3]
###### Training configs ######
NSTEP=35000
DECAY=0.98

MAX_ITER=3000 # defines the size of an epoch
SNAPSHOT_INTERVAL=1000 # interval for saving snapshot
SEED=2025

N_PART=3 # defines the number of chunks for evaluation
ALL_SUPP=(2) # CHAOST2: 0-4, CMR: 0-7
model_id=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000 105000 110000 115000 120000 125000 130000 135000 140000 145000 150000 155000 160000 165000 170000 175000 180000 185000 190000 195000)
output_file="results.txt" 
echo ========================================================================
for id in "${model_id[@]}"
do
  sum=0
  for EVAL_FOLD in "${ALL_EV[@]}"
  do
     PREFIX="test_${DATASET}_cv${EVAL_FOLD}"
     echo $PREFIX
     LOGDIR="./results"

     if [ ! -d $LOGDIR ]
     then
        mkdir -p $LOGDIR
     fi
     for SUPP_IDX in "${ALL_SUPP[@]}"
     do
        # RELOAD_PATH='please feed the absolute path to the trained weights here' # path to the reloaded model
        RELOAD_MODEL_PATH="model.pth"
        python3 test.py with \
        mode="test" \
        dataset=$DATASET \
        num_workers=$NWORKER \
        n_steps=$NSTEP \
        eval_fold=$EVAL_FOLD \
        max_iters_per_load=$MAX_ITER \
        supp_idx=$SUPP_IDX \
        test_label=$TEST_LABEL \
        seed=$SEED \
        n_part=$N_PART \
        reload_model_path=$RELOAD_MODEL_PATH \
        save_snapshot_every=$SNAPSHOT_INTERVAL \
        lr_step_gamma=$DECAY \
        path.log_dir=$LOGDIR
    done

    value=$(<results.txt)
    #echo "累加结果是: $value"
    sum=$(echo "$sum + $value" | bc)

  done
  sum=$(echo "scale=5; $sum / 5" | bc)
  echo "result of ${id} is: $sum"
  echo -e "\n$sum" >> "$output_file"
done


