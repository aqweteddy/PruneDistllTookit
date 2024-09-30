from typing import Union, Optional, Dict, Any
from transformers import PreTrainedModel
import torch, random
from torch import nn
from trl import GKDConfig, GKDTrainer
from trl.trainer.utils import empty_cache
from trl.models.utils import unwrap_model_for_generation

class CustomKDTrainer(GKDTrainer):
    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        args: Optional[GKDConfig] = None,
        loss_modules: list[str] = ['mlp_output'], # ['embed', 'logits', 'mlp_output']
        teacher_student_layer_map: Optional[dict[int, int]] = None,
        *sft_args,
        **kwargs,
    ):
        super().__init__(teacher_model, args, *sft_args, **kwargs)
        self.loss_modules = loss_modules
        self.teacher_student_layer_map = teacher_student_layer_map
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        return loss
    
    @staticmethod
    def layer_mse_loss(
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(student_hidden, teacher_hidden)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )

        # compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            input_ids = inputs["input_ids"] # [batch_size, seq_len]
            # labels = inputs["labels"] # [batch_size, seq_len]
            # input_ids_labels = torch.cat([input_ids, labels], dim=-1)
            outputs_teacher = self.teacher_model(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

        # slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]

        # compute loss
        loss = 0.
        if "logits" in self.loss_modules:
            logits_loss = self.generalized_jsd_loss(
                student_logits=shifted_student_logits,
                teacher_logits=shifted_teacher_logits,
                labels=shifted_labels,
                beta=self.beta,
            )
            loss += logits_loss
        
        if "embed" in self.loss_modules:
            student_embeds = outputs_student.hidden_states[0]
            teacher_embeds = outputs_teacher.hidden_states[0]
            embed_loss = self.layer_mse_loss(student_embeds, teacher_embeds)
            loss += embed_loss
        
        if 'mlp_output' in self.loss_modules:
            for teacher_layer_idx, student_layer_idx in self.teacher_student_layer_map.items():
                teacher_layer = outputs_teacher.hidden_states[teacher_layer_idx]
                student_layer = outputs_student.hidden_states[student_layer_idx]
                mlp_output_loss = self.layer_mse_loss(student_layer, teacher_layer)
                loss += mlp_output_loss
        
        # empty cache
        empty_cache()

        # Return loss
        return (loss, outputs_student) if return_outputs else loss