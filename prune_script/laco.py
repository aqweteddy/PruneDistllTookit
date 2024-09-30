from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch import nn
import torch
from tqdm import tqdm
import copy
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
    def __substrct(a: dict, b: dict):
        new_params = {}
        for k, v in a.items():
            new_params[k] = v - b[k]
        return new_params
    def __add(a: dict, b: dict):
        new_params = {}
        for k, v in a.items():
            new_params[k] = v + b[k]
        return new_params

    result = base_l.state_dict()
    for l in l_s:
        delta = __substrct(l.state_dict(), base_l.state_dict())
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
    for batch in tqdm(samples, leave=False):
        # to cuda
        for k, v in batch.items():
            batch[k] = v.to('cuda')
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        last_hidden  = model(**batch).hidden_states[-1] # [batch_size, hidden_size]
        for batch in range(last_hidden.size(0)):
            last_non_pad_index = attention_mask[batch].nonzero(as_tuple=True)[0].max()
            
            result.append(last_hidden[batch, last_non_pad_index, :].unsqueeze(0))
    
    return torch.cat(result, dim=0)

def get_cosine_similarity_of_layers(
    hidden1: torch.Tensor,
    hidden2: torch.Tensor,
):
    hidden1 = hidden1 / torch.norm(hidden1, dim=-1, keepdim=True)
    hidden2 = hidden2 / torch.norm(hidden2, dim=-1, keepdim=True)
    cosine_similarity = (hidden1 * hidden2).sum(-1)
    print(cosine_similarity)
    return cosine_similarity.mean()

def fix_layer_index(
    model: LlamaForCausalLM,
):
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i
        
    return model

def laco(
    model: LlamaForCausalLM,
    C: int, # Number of layers combined in each merge
    I: int, # Minimum interval between two adjacent merged layers
    original_hidden: torch.Tensor, # Hidden states of the original model
    samples: DataLoader,
    T: float, # Threshold for representation similarity
    layer_range: tuple[int, int], # Range of layers to be merged
):
    M: nn.ModuleList[LlamaDecoderLayer] = model.model.layers
    print('original_M', len(M))
    l = layer_range[1] - C # from top
    while l >= layer_range[0]:
        k = min(C - 1, len(M) - l)
        print(f'{k=}, {l=}, {len(M)=}')
        logging.info(f"Merge layers from {l} to {l+k}")
        new_layer = layer_merge(M[l], M[l+1:l+k]) # merge layers from l to k
        tmp_M = copy.deepcopy(M[:l]) + [new_layer] + copy.deepcopy(M[l+k:])
        print('tmp_M', len(tmp_M))
        
        model.model.layers = nn.ModuleList(tmp_M)
        
        # calculate the similarity
        model = fix_layer_index(model)
        tmp_hidden = inference_last_hidden(model, samples)
        sim = get_cosine_similarity_of_layers(original_hidden, tmp_hidden)
        
        if sim >= T:
            logging.info(f"Similarity: {sim}, Merge layers from {l} to {l+k}")
            M = tmp_M
            l -= I
            original_hidden = tmp_hidden
            if l > len(M) - C:
                l = len(M) - C
        else:
            logging.info(f"Similarity: {sim}, Skip merging layers from {l} to {l+k}")
            l -= 1
    
    model.model.layers = nn.ModuleList(M)
    return model


def apply_chat_template(dct: dict, col: str, tokenizer: AutoTokenizer):
    res = tokenizer.apply_chat_template(dct[col], tokenize=False)
    
    return {'_mes_str': res}

def count_parameters(model: LlamaForCausalLM):
    return sum(p.numel() for p in model.parameters())


def main(
    model_path: str = '/volume/models/Qwen/Qwen2.5-1.5B-Instruct/',
    dataset: dict = "{'path':'aqweteddy/mrc','revision':'v0_cite'}",
    dataset_size: int=100,
    output_path: str = "/volume/models/test/",
    batch_size: int = 2,       
    C: int = 4, # Number of layers combined in each merge
    I: int = 2, # Minimum interval between two adjacent merged layers
    T: float = 0.6, # Threshold for representation similarity
):
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', output_hidden_states=True)
    model.to('cuda')
    logging.info(f"Model has {count_parameters(model)} parameters")
    
    config: LlamaConfig = model.config
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = datasets.load_dataset(**dataset)['train'] 
    if dataset_size:
        dataset = dataset.shuffle().select(range(dataset_size))
    
    # tokenizer.padding_side = 'left'
    dataset = dataset.map(apply_chat_template, fn_kwargs={'tokenizer': tokenizer, 
                                                          'col': 'messages'},
                          num_proc=8)
    
    inputs = tokenizer(dataset['_mes_str'], return_tensors="pt", padding="longest", truncation=True)
    inputs = [{'input_ids': inputs['input_ids'][i], 'attention_mask': inputs['attention_mask'][i]} for i in range(len(inputs['input_ids']))]
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
    logging.info(f"Model has {count_parameters(model)} parameters after pruning.")
    model.save_pretrained(output_path, max_shard_size='8GB')
    tokenizer.save_pretrained(output_path)
    

if __name__ == '__main__':
    Fire(main)