from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from transformers import Trainer, is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

class DistillRankingTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, teacher=None, loss=None, batch_size=8, max_sequence_length=384,distill_hardness =0.5, temperature=2.0, criterion=nn.CrossEntropyLoss(), **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.loss = loss
        self.teacher = teacher
        self.batch_size = batch_size
        self.temperature = temperature
        self.distill_hardness = distill_hardness
        self.criterion = criterion # coult be nn.KLDivLoss() or nn.MSELoss() or nn.CrossEntropyLoss(), 
        self.max_sequence_length = max_sequence_length
        if self.teacher is None:
            self.distill_hardness = 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. Modified for Distilation using student teacher framework modified for distilation. 
        """
        input_device = inputs["input_ids"].device
        outputs = model(**inputs)
        start_logits_student = outputs["start_logits"]
        end_logits_student = outputs["end_logits"]
        start_logits_label = inputs["start_positions"]
        end_logits_label = inputs["start_positions"]
        if teacher is not None:
            self.teacher = self.teacher.to(input_device)
            with torch.no_grad():
                teacher_output = self.teacher(
                                input_ids=inputs["input_ids"],
                                token_type_ids=inputs["token_type_ids"],
                                attention_mask=inputs["attention_mask"],
                            )
            start_logits_teacher = teacher_output["start_logits"]
            end_logits_teacher = teacher_output["end_logits"]
            loss_start = (
                F.kl_div(
                    input=F.log_softmax(start_logits_student / self.temperature, dim=-1),
                    target=F.softmax(start_logits_teacher / self.temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.temperature ** 2)
            )
            loss_end = (
                F.kl_div(
                    input=F.log_softmax(end_logits_student / self.temperature, dim=-1),
                    target=F.softmax(end_logits_teacher / self.temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.temperature ** 2)
            )
            teacher_loss = (loss_start + loss_end) / 2.0
        loss_start = self.criterion(start_logits_student, start_logits_label)
        loss_end = self.criterion(end_logits_student, end_logits_label)
        label_loss = (loss_start + loss_end) / 2.0
        loss = ((1-self.distill_hardness) * label_loss) + (self.distill_hardness * teacher_loss)
        return loss    