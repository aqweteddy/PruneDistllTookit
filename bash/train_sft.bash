#!/bin/bash

python train_script/sft.py \
--model_name_or_path /volume/models/Yi-1.5-9B-Chat \
--torch_dtype bfloat16 \
--dataset "{'path': 'aqweteddy/mrc','revision':'v0'}"
--packing \
--dataset_num_proc 8 \
--use_liger \
