import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Module, Parameter
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm
import math
import wandb

logger = logging.getLogger(__name__)

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_datasets_available,
    is_torch_tpu_available,
    set_seed,
)

from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.trainer_utils import PredictionOutput, is_main_process

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.optim.optimizer import ScheduledOptimizer
from sparseml.pytorch.utils import ModuleExporter

from distill_trainer import DistillRankingTrainer

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default='data/train.json', metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default='data/evaluation.json', metadata={"help": "A json file containing the validation data."}
    )
    nm_prune_config: Optional[str] = field(
        default='recipes/base.yaml', metadata={"help": "The input file name for the Neural Magic pruning config"}
    )
    max_train_samples: Optional[int] = field(
        default=8000000,
        metadata={
            "help": "Since the MSMARCO Dataset is 79551622 items we subsample to ~10%"
        },
    )
    do_onnx_export: bool = field(
        default=False, metadata={"help": "Export model to onnx"}
    )
    onnx_export_path: Optional[str] = field(
        default='onnx-export', metadata={"help": "The filename and path which will be where onnx model is outputed"}
    )
    layers_to_keep: int = field(
        default=12, metadata={"help":"How many layers to keep for the model"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    teacher_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Teacher model which needs to be a trained sequence classification model"}
    )
    student_model_name_or_path: Optional[str] = field(
        default="bert-base-uncased", metadata={"help": "Student model"}
    )
    temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Temperature applied to teacher softmax for distillation."}
    )
    distill_hardness: Optional[float] = field(
        default=1.0, metadata={"help": "Proportion of loss coming from teacher model."}
    )
    config_name: Optional[str] = field(
        default='bert-base-uncased', metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default='bert-base-uncased', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='cache',
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

def load_ranking(filename, collection, queries):
    qid2documents = {}
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            query = queries[l[0]]
            document = collection[l[1]]
            if query not in qid2documents:
                qid2documents[query] = []
            qid2documents[query].append(document)
    return qid2documents

def load_qid2query(filename):
    qid2query = {}
    with open(filename,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            qid2query[int(l[0])] = l[1]
    return qid2query

def load_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer_cls = AdamW
    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = args.learning_rate
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

def drop_layers(model, layers_to_keep):
    layer_drop_matching = {
        1:[0],
        3:[0,5,11],
        6:[0,2,4,6,8,11],
        9:[0,2,3,4,5,7,8,9,11],
        12:[0,1,2,3,4,5,6,7,8,9,10,11],
    }
    encoder_layers = model.bert.encoder.layer # change based on model name
    assert layers_to_keep <= len(encoder_layers)
    assert layers_to_keep in layer_drop_matching.keys()
    trimmed_encoder_layers = nn.ModuleList()
    for i in layer_drop_matching[layers_to_keep]:
        trimmed_encoder_layers.append(encoder_layers[i])
    trimmed_model = copy.deepcopy(model)
    trimmed_model.bert.encoder.layer = trimmed_encoder_layers
    return trimmed_model


def main():
    wandb.init(project='PruneMSMARCO', entity='spacemanidol')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    label_list = datasets["train"].unique("label")
    label_list.sort()
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task='msmarco-triples',
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    student_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.student_model_name_or_path,
        from_tf=bool(".ckpt" in model_args.student_model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    if data_args.layers_to_keep < len(student_model.bert.encoder.layer):
        logger.info("Keeping %s model layers", data_args.layers_to_keep)
        student_model = drop_layers(student_model, data_args.layers_to_keep)
    if model_args.teacher_model_name_or_path != None:
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.teacher_model_name_or_path,
            from_tf=bool(".ckpt" in model_args.teacher_model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        teacher_model_parameters = filter(lambda p: p.requires_grad, teacher_model.parameters())
        params = sum([np.prod(p.size()) for p in teacher_model_parameters])
        logger.info("Teacher Model has %s parameters", params)  
    else:
        teacher_model = None

    student_model_parameters = filter(lambda p: p.requires_grad, student_model.parameters())
    params = sum([np.prod(p.size()) for p in student_model_parameters])
    logger.info("Student Model has %s parameters", params)  

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples["query"], examples["passage"])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        result["label"] = examples["label"]
        return result
    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
    if data_args.max_train_samples is not None:
        datasets["train"] = datasets["train"].select(range(data_args.max_train_samples))
    traindataset = datasets["train"]
    for index in random.sample(range(len(traindataset)), 3):
        logger.info(f"Sample {index} of the training set: {traindataset[index]}.")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    optim = load_optimizer(student_model, TrainingArguments)
    steps_per_epoch = math.ceil(len(datasets["train"]) / (training_args.per_device_train_batch_size*training_args._n_gpu))
    manager = ScheduledModifierManager.from_yaml(data_args.nm_prune_config)
    optim = ScheduledOptimizer(optim, student_model, manager, steps_per_epoch=steps_per_epoch, loggers=None)
    training_args.num_train_epochs = float(manager.modifiers[0].end_epoch)

    trainer = DistillRankingTrainer(
        model=student_model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optim, None),
        teacher=teacher_model,
        distill_hardness = model_args.distill_hardness,
        temperature = model_args.temperature,
    )
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()