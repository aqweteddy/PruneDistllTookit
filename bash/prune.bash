python prune_script/prune_layer.py \
--model_path /volume/models/Yi-1.5-9B-Chat/ \
--layers_to_skip 16 \
--dataset '{"path":"aqweteddy/mrc","revision":"v0_cite"}' \
--dataset_size 10000 \
--device cuda