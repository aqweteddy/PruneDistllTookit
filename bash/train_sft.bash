#!/bin/bash
accelerate launch --config_file deepspeed_cfg/zero2.yml --num_processes 4 train_script/sft.py \
--model_name_or_path /volume/models/Qwen/Qwen2.5-1.5B-Instruct \
--torch_dtype bfloat16 \
--output_dir /volume/models/rag_ckpt/test \
--dataset_name "{'path': 'aqweteddy/mrc','revision':'v0'}" \
--dataset_num_proc 8 \
--use_liger true \
--learning_rate 1e-5 \
--weight_decay 0.01 \
--optim paged_adamw_8bit \
--lr_scheduler_type cosine \
--warmup_steps 100 \
--per_device_train_batch_size 1 \
--max_length 4096 \
--num_train_epochs 3 \
--gradient_checkpointing \
--report_to wandb \
--save_only_model \
--save_safetensors 
