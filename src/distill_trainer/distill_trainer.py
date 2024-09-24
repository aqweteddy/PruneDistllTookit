from typing import Union, Optional
from transformers import PreTrainedModel
import torch
from torch import nn
from trl import GKDConfig, GKDTrainer
from trl.trainer.utils import empty_cache


class CustomGKDTrainer(GKDTrainer):
    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        args: Optional[GKDConfig] = None,
        loss_modules: list[str] = ['logits'], # ['embed', 'logits', 'mlp_input', 'mlp_output']
        teacher_student_layer_map: Optional[dict[int, int]] = None,
        *sft_args,
        **kwargs,
    ):
        super().__init__(teacher_model, args, *sft_args, **kwargs)
        self.loss_modules = loss_modules
        self.teacher_student_layer_map = teacher_student_layer_map
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]

        # compute loss
        #!TODO
        # for loss_type in self.loss_modules:
        #     if 
        
        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
        )

        # empty cache
        empty_cache()

        # Return loss
        return (loss, outputs_student) if return_outputs else loss