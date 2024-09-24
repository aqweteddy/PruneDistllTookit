from trl.commands.cli_utils import SFTScriptArguments, TrlParser
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from bitsandbytes.optim import AdamW8bit

from datasets import load_dataset

from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model init kwargs & Tokenizer
    ################
    
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation='flash_attention_2',
        torch_dtype=model_config.torch_dtype,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, 
        trust_remote_code=model_config.trust_remote_code, 
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset(**eval(args.dataset_name))['train']
    dataset = dataset.train_test_split(test_size=1000)

    ################
    # Training
    ################
    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        **model_kwargs,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)