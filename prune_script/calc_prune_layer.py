import logging
import csv
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets

from typing import Optional

logging.basicConfig(level=logging.INFO)

# Set seed
torch.manual_seed(42)
np.random.seed(42)

def angular_distance(x_l, x_l_plus_n) -> torch.Tensor:
    """Compute the angular distance between layer output tokens."""
    x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True)
    x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True)
    cosine_similarity = (x_l_norm * x_l_plus_n_norm).sum(-1)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi

def compute_block_distances(hidden_states: list[torch.Tensor], layers_to_skip: int) -> list[float]:
    """Compute and return angular distances for each block of layers."""
    distances = []
    num_layers = len(hidden_states)
    for l in range(num_layers - layers_to_skip):
        block_distance = angular_distance(hidden_states[l], hidden_states[l + layers_to_skip]).mean().item()
        distances.append(block_distance)
    return distances

def get_last_non_padded_tokens(hidden_states, attention_mask) -> list[torch.Tensor]:
    """Get last non-padded tokens for each layer."""
    last_non_padded_hidden_states = []
    for layer in hidden_states:
        batch_size, _, _ = layer.size()
        batch_last_tokens = []
        for batch in range(batch_size):
            last_non_pad_index = attention_mask[batch].nonzero(as_tuple=True)[0].max()
            last_token = layer[batch, last_non_pad_index, :]
            batch_last_tokens.append(last_token.unsqueeze(0))
        last_non_padded_hidden_states.append(torch.cat(batch_last_tokens, dim=0))
    return last_non_padded_hidden_states

def apply_chat_template(dct: dict, col: str, tokenizer: AutoTokenizer):
    res = tokenizer.apply_chat_template(dct[col], tokenize=False)
    return {'_mes_str': res}

def main(model_path: str, dataset: str, dataset_column: str, batch_size: int, max_length: int,
         layers_to_skip: int, dataset_size: Optional[int] = None, dataset_subset: Optional[str] = "eval"):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    model = AutoModelForCausalLM.from_pretrained(model_path,  
                                                 device_map="auto", 
                                                 attn_implementation="flash_attention_2",
                                                 output_hidden_states=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    dataset = datasets.load_dataset(**eval(dataset))[dataset_subset] 
    if dataset_size:
        dataset = dataset.select(range(dataset_size))
    dataset = dataset.map(apply_chat_template, fn_kwargs={'tokenizer': tokenizer, 
                                                          'col': dataset_column},
                          num_proc=8)
    dataloader = DataLoader(dataset['_mes_str'], batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize a list to store distances for each block across the dataset
    all_distances = [[] for _ in range(model.config.num_hidden_layers - layers_to_skip)]


    for batch in tqdm(dataloader, desc="Processing batches"):
        inputs = tokenizer(batch, return_tensors="pt", padding="longest", max_length=max_length, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.hidden_states
        last_non_padded_hidden_states = get_last_non_padded_tokens(hidden_states, attention_mask)

        # Remove the first element to account for the input layer not being considered a model hidden layer
        # This adjustment is necessary for analyses focusing on the model's internal transformations
        last_non_padded_hidden_states = last_non_padded_hidden_states[1:]
        
        # Ensure that the length of last_non_padded_hidden_states matches the number of model hidden layers minus one
        assert len(last_non_padded_hidden_states) == model.config.num_hidden_layers, "Length of last_non_padded_hidden_states  \
        does not match expected number of hidden layers."

        # Compute distances and append to all_distances
        distances = compute_block_distances(last_non_padded_hidden_states, layers_to_skip)
        for i, distance in enumerate(distances):
            all_distances[i].append(distance)

    # Calculate average distances for each block
    average_distances = [np.mean(block_distances) for block_distances in all_distances]

    # Write the average distances to a CSV file and compute the minimum average distance
    min_distance = float('inf')  # Initialize with infinity
    min_distance_layer = 0  # Initialize with an impossible value

    layers = []
    for i, avg_dist in enumerate(average_distances):
        layers.append({
            'block_start': i,
            'block_end': i + layers_to_skip,
            'distance': avg_dist
        })
    
    layers = sorted(layers, key=lambda x: x['distance'])
    for layer in layers:
        logging.info(f"Block {layer['block_start']} to {layer['block_end']}: {layer['distance']}")
    
    # write to csv
    with open(f"{model_path}/{dataset}_layer_distances.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=['block_start', 'block_end', 'distance'])
        writer.writeheader()
        for layer in layers:
            writer.writerow(layer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model analysis.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--dataset_column", type=str, help="The specific column of the dataset to use.", default='messages')
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing.")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum length of the tokenized input.")
    parser.add_argument("--layers_to_skip", type=int, required=True, help="Number of layers to skip.")
    parser.add_argument("--dataset_size", type=int, help="Optional argument to specify the size of the dataset.")
    parser.add_argument("--dataset_subset", type=str, default="train", help="Subset of the dataset to use (e.g., 'train', 'eval').")
    parser.add_argument("--device", type=str, help="Device to run the model on ('cpu', 'cuda').")

    args = parser.parse_args()

    main(args.model_path, args.dataset, args.dataset_column, args.batch_size,
         args.max_length, args.layers_to_skip, args.dataset_size, args.dataset_subset)
