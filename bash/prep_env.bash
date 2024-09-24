# pip install -U 'xtuner[deepspeed]'
pip install flash-attn --no-build-isolation
pip install wandb hf_transfer
pip install -r requirements.txt
pip install git+github.com/arcee-ai/mergekit.git
apt update
apt install vim nvtop tmux -y
export HF_DATASETS_CACHE="/workspace/.cache/hf/dataset"

wandb login