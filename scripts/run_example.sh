TRAIN_FILE_PATH=/your/path/to/train.txt
EVAL_FILE_PATH=/your/path/to/eval.txt
OUTPUT_DIR=../checkpoints/
RUN_PATH=../runs/
MODEL_PATH=../checkpoints/xxx/
# python main.py \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 main.py \
    --dataset_name DPCFv5 \
    --src_pretrain_dataset_name "bert-base-chinese" \
    --tgt_pretrain_dataset_name "gpt2" \
    --train_data_path $TRAIN_FILE_PATH \
    --eval_data_path $EVAL_FILE_PATH \
    --output_dir $OUTPUT_DIR\
    --run_path $RUN_PATH \
    --finetune_lr 3e-5 \
    --lr 1e-4 \
    --num_training_steps 150000 \
    --num_warmup_steps 10000 \
    --max_src_len 128 \
    --max_tgt_len 128 \
    --save_step 1 \
    --nepoch 3 \
    --ngpu 1 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    # --drop_last \
    # --model_path $MODEL_PATH \
    # --ispredict \
