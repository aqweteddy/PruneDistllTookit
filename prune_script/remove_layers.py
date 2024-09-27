from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaConfig
from torch import nn
import json
from fire import Fire


def main(
    model_path: str,
    output_path: str = "ckpt/pruned_model",
    remove_ids: list[int] = [],
):
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    config: LlamaConfig = model.config
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    new_layers = []
    config.num_hidden_layers = config.num_hidden_layers - len(remove_ids)
    
    # from 0
    new_idx = 0
    for i, layer in enumerate(model.model.layers):
        if i not in remove_ids:
            layer.self_attn.layer_idx = new_idx
            new_idx += 1
            new_layers.append(layer)
        else:
            print(f"Removed layer {i}")
    decoder_layers = nn.ModuleList(new_layers)
    model.model.layers = decoder_layers
    
    model.save_pretrained(output_path, max_shard_size='8GB')
    tokenizer.save_pretrained(output_path)
    
    with open(f"{output_path}/pruning.json", "w") as f:
        json.dump({"removed_layers": remove_ids}, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    Fire(main)