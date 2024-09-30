from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch import nn
import torch
from tqdm import tqdm
import datasets

import json, logging
from torch.utils.data import DataLoader
from fire import Fire


datasets.disable_caching()
logging.basicConfig(level=logging.INFO)

def layer_merge(
    base_l: LlamaDecoderLayer,
    l_s: list[LlamaDecoderLayer],
):
    """
    Merge layers by adding them together.
    
    """
    def __substrct(a: LlamaDecoderLayer, b: LlamaDecoderLayer):
        new_params = {}
        for k, v in a.state_dict().items():
            new_params[k] = v - b.state_dict()[k]
        return new_params
    def __add(a: LlamaDecoderLayer, b: LlamaDecoderLayer):
        new_params = {}
        for k, v in a.state_dict().items():
            new_params[k] = v + b.state_dict()[k]
        return new_params

    result = base_l.state_dict()
    for l in l_s:
        delta = __substrct(l, base_l)
        result = __add(result, delta)
    
    base_l.load_state_dict(result)
    return base_l

@torch.inference_mode()
def inference_last_hidden(
    model: LlamaForCausalLM,
    samples: DataLoader,
):
    model.eval()
    result = []
    with torch.no_grad():
        for batch in tqdm(samples):
            last_hidden  = model(**batch).last_hidden_state[:, -1, :] # [batch_size, hidden_size]
            result.append(last_hidden.detach().cpu())
    return torch.cat(result, dim=0)

def get_cosine_similarity_of_layers(
    hidden1: torch.Tensor,
    hidden2: torch.Tensor,
):
    hidden1 = hidden1 / torch.norm(hidden1, dim=-1, keepdim=True)
    hidden2 = hidden2 / torch.norm(hidden2, dim=-1, keepdim=True)
    cosine_similarity = (hidden1 * hidden2).sum(-1)
    return cosine_similarity.mean()


def laco(
    model: LlamaForCausalLM,
    c: int, # Number of layers combined in each merge
    I: int, # Minimum interval between two adjacent merged layers
    original_hidden: torch.Tensor, # Hidden states of the original model
    samples: DataLoader,
    T: float, # Threshold for representation similarity
    layer_range: tuple[int, int], # Range of layers to be merged
):
    M: nn.ModuleList[LlamaDecoderLayer] = model.model.layers
    
    l = len(layer_range[1]) - c # from top
    while l >= layer_range[0]:
        k = min(c-1, len(M) - l)
        
        logging.info(f"Merge layers from {l} to {l+k}")
        new_layer = layer_merge(M[l], M[l+1:k])
        tmp_M = M[:l] + new_layer + M[l+k:]
        model.model.layers = nn.ModuleList(tmp_M)
        
        # calculate the similarity
        tmp_hidden = inference_last_hidden(model, samples)
        sim = get_cosine_similarity_of_layers(original_hidden, tmp_hidden)
        
        if sim >= T:
            logging.info(f"Similarity: {sim}, Merge layers from {l} to {l+k}")
            M = tmp_M
            l -= I
            if l > len(M) - c:
                l = len(M) - c
        else:
            logging.info(f"Similarity: {sim}, Skip merging layers from {l} to {l+k}")
            l -= 1
    
    model.model.layers = nn.ModuleList(M)
    return model


def apply_chat_template(dct: dict, col: str, tokenizer: AutoTokenizer):
    res = tokenizer.apply_chat_template(dct[col], tokenize=False)
    
    return {'_mes_str': res}

def count_parameters(model: LlamaForCausalLM):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(
    model_path: str = '/volume/models/Qwen/Qwen2.5-1.5B-Instruct/',
    dataset: str = "{'path':'aqweteddy/mrc','revision':'v0_cite'}",
    dataset_size: int=100,
    output_path: str = "/volume/models/test/",
    batch_size: int = 2,       
    C: int = 2, # Number of layers combined in each merge
    I: int = 2, # Minimum interval between two adjacent merged layers
    T: float = 0.9, # Threshold for representation similarity
):
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    logging.info(f"Model has {count_parameters(model)} parameters")
    
    config: LlamaConfig = model.config
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = datasets.load_dataset(**eval(dataset))['train'] 
    if dataset_size:
        dataset = dataset.shuffle().select(range(dataset_size))
    
    tokenizer.padding_side = 'left'
    dataset = dataset.map(apply_chat_template, fn_kwargs={'tokenizer': tokenizer, 
                                                          'col': 'messages'},
                          num_proc=8)
    
    inputs = tokenizer(dataset['_mes_str'], return_tensors="pt", padding="longest", truncation=True)
    samples = DataLoader(inputs, 
                         batch_size=batch_size, 
                         shuffle=False, 
                         drop_last=False)
    
    model = laco(
            model=model,
            C=C,
            I=I,
            original_hidden=inference_last_hidden(model, samples),
            samples=samples,
            T=T,
            layer_range=(1, len(model.model.layers) - 1),
    )
    config.num_hidden_layers = len(model.model.layers)
    tokenizer.padding_side = 'right'
    logging.info(f"Model has {count_parameters(model)} parameters after pruning.")
    model.save_pretrained(output_path, max_shard_size='8GB')
    tokenizer.save_pretrained(output_path)
    

if __name__ == '__main__':
    Fire(main)