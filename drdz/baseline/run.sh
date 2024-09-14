#!/bin/bash

log_path="./outputs/lightning_logs/"
# ===== Set your dataset path here =====
train_data_path="./data/first_stage_train_5000.jsonl"
val_data_path=""
test_data_path="./data/first_stage_test_300.jsonl"

do_train=true
# provided test set has no ground truth, do_test will fail
do_test=false
model_load_path="none"
epochs=16
# batch size per GPU! actual batch size = batch_size * num_gpus
batch_size=32
accumulate_grad_batches=1
num_workers=4
lr=2e-5

# ===== Set your model load path here. Required if `do_test` and not `do_train` =====
model_load_path=""


params=""

if [ "$do_train" = true ]; then
  params="$params --do_train"
fi

if [ "$do_test" = true ]; then
  params="$params --do_test"
fi

if [ "$val_data_path" != "" ]; then
  params="$params --val_data_path $val_data_path"
fi

if [ "$model_load_path" != "" ]; then
  params="$params --model_load_path $model_load_path"
fi
# echo "params: $params"

python3 -m bert.main \
    --train_data_path $train_data_path \
    --test_data_path $test_data_path \
    --model_load_path $model_load_path \
    --log_path $log_path \
    --accumulate_grad_batches $accumulate_grad_batches \
    --epochs $epochs \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --lr $lr \
    $params | tee "./log/bert.log"
  