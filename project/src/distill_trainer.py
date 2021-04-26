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

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions
            )
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        eval_preds = self.post_process_function(
            test_examples, test_dataset, output.predictions
        )
        metrics = self.compute_metrics(eval_preds)

        return PredictionOutput(
            predictions=eval_preds.predictions,
            label_ids=eval_preds.label_ids,
            metrics=metrics,
        )
    
    