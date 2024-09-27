python prune_script/prune_layer.py \
--model_path /volume/models/rag_slm_ckpt/yi-1.5-9b-chat_ft-mrc-v0-cite-s1533/ \
--layers_to_skip 18 \
--dataset '{"path":"aqweteddy/mrc","revision":"v0_cite"}' \
--batch_size 2 \
--dataset_size 10000 \
--device cuda
