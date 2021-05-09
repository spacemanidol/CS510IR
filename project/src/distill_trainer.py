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
    def __init__(self, *args, eval_examples=None, post_process_function=None, teacher=None, loss=None, batch_size=64, max_sequence_length=128,distill_hardness =0.5, temperature=2.0, criterion=nn.CrossEntropyLoss(), **kwargs):
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
        loss = outputs['loss']
        if self.teacher is not None:
            self.teacher = self.teacher.to(input_device)
            student_logit_neg = outputs['logits'][:, :1]
            student_logit_pos = outputs['logits'][:, 1:2]
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs)
                teacher_logit_neg = teacher_outputs['logits'][:, :1]
                teacher_logit_pos = teacher_outputs['logits'][:, 1:2]
            loss_pos = F.kl_div( input=student_logit_pos, target=teacher_logit_pos, reduction="batchmean",) * (self.temperature ** 2)
            loss_neg = F.kl_div( input=student_logit_neg, target=teacher_logit_neg, reduction="batchmean",) * (self.temperature ** 2)
            teacher_loss = torch.stack([loss_pos,loss_neg])
            loss = ((1-self.distill_hardness) * loss) + torch.abs(((self.distill_hardness * teacher_loss)))
        return (loss, outputs) if return_outputs else loss       