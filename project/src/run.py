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

config = AutoConfig.from_pretrained('cache/base')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained('cache/base')
   
eval_dataset = load_dataset("glue", "qqp", split='test')
sentence1_key, sentence2_key = "question1", "question2"
padding = 'max_length'
max_seq_length = 128

def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    result["label"] = [0 if l == -1 else l for l in examples["label"]]
    return result

eval_dataset = eval_dataset.map(preprocess_function, batched=True)
print(len(eval_dataset))
eval_data = eval_dataset.select(range(16))
trainer = Trainer(model=model,)
predictions = trainer.predict(test_dataset=eval_dataset.select(range(16)))
print(predictions)
"""
def load_qid2query(filename):
    qid2query = {}
    with open(filename,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            qid2query[int(l[0])] = l[1]
    return qid2query

queries = load_qid2query('data/queries.tsv')
collection = load_qid2query('data/collection.tsv')


def load_ranking(filename, collection, queries):
    qid2documents = {}
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            query = queries[int(l[0])]
            document = collection[int(l[1])]
            if query not in qid2documents:
                qid2documents[query] = []
            qid2documents[query].append(document)
    return qid2documents


ranking = load_ranking('data/bm25devtop1000.txt', collection, queries)
query = list(ranking.keys())[0]
candidates = ranking[query]
data = tokenizer([query for i in range(len(candidates))], candidates, padding=False)[:8]
print("loaded data")
print(data)
trainer = Trainer(
        model=model,
        eval_dataset=data
    )

print(trainer.evaluate())
"""