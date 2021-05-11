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
        outputs = model(**inputs)
        loss = outputs["loss"]
        logits_student = outputs["logits"]
        if self.teacher is not None:
            input_device = inputs["input_ids"].device
            self.teacher = self.teacher.to(input_device)
            with torch.no_grad():
                teacher_outputs = self.teacher(
                        input_ids=inputs["input_ids"],
                        token_type_ids=inputs["token_type_ids"],
                        attention_mask=inputs["attention_mask"],
                )
            logits_teacher = teacher_outputs["logits"]
            loss_distill = F.kl_div( input=logits_student, target=logits_teacher, reduction="batchmean",) * (self.temperature ** 2)
            loss = ((1-self.distill_hardness) * loss) + torch.abs((self.distill_hardness * loss_distill))
        return (loss, outputs) if return_outputs else loss
