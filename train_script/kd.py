from trl.commands.cli_utils import TrlParser
from trl import GKDConfig, GKDTrainer, SFTScriptArguments
from liger_kernel.transformers import AutoLigerKernelForCausalLM

from datasets import load_dataset

from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
)

from distill_trainer.distill_trainer import CustomKDTrainer


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, GKDConfig, ModelConfig))
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
    student_model = AutoLigerKernelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        **model_kwargs,
    )
    teacher_model = AutoLigerKernelForCausalLM.from_pretrained(
        training_args.teacher_model_name_or_path,
        **model_kwargs,
    )
    training_args.max_new_tokens = 2048 if training_args.max_new_tokens == 128 else training_args.max_new_tokens

    trainer = CustomKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)