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
sentence1_key, sentence2_key = "query", "passage"
padding = 'max_length'
max_seq_length = 128

def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    return result

dataset = dataset.map(preprocess_function, batched=True)
print(len(eval_dataset))
eval_data = eval_dataset.select(range(16))
trainer = Trainer(model=model,)
predictions = trainer.predict(test_dataset=eval_dataset.select(range(16))).predictions

def main(args):
    sentence1_key, sentence2_key = "query", "passage"
    padding = 'max_length'
    max_seq_length = args.max_seq_length
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
files = {"predict":args.dataset_name)
dataset = load_dataset("json", data_files=files, cache_dir=args.cache_dir)
idx2qidpid = {}
for i in range(dataset['predict'].shape[0]):
    idx2qidpid[i] = (dataset['predict'][i]['qid'],dataset['predict'][i]['pid'])
def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    return result
dataset = dataset.map(preprocess_function, batched=True)
trainer = Trainer(model=model,)
predictions = trainer.predict(test_dataset=dataset["predict"]).predictions
    qid2rank = {}
    for i in range(predictions.shape[0]):
        qid = 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn MSMARCO Data into HF usable datasets')
    parser.add_argument('--model_name_or_path', type=str, default='cache/base', help="reranking model")
    parser.add_argument('--cache_dir', type=str, default='cache', help='cache file')
    parser.add_argument('--max_seq_length', type=int, default=256, help="padding size")
    parser.add_argument('--dataset_name', type=str, default='devtop1000.json', help='json processed results')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='bert-base-uncased', help='tokenizer')
    parser.add_argument('--ranking_file', type=str, default='data/bm25devtop1000.txt', help='BM25 Ranked file')
    parser.add_argument('--query_file', type=str, default='data/queries.tsv', help="qid 2 query file")
    parser.add_argument('--collection_file', type=str, default='data/collection.tsv', help='docid 2 doc')
    parser.add_argument('--output_filename', type=str, default='devtop1000.json', help='Output file')
    args = parser.parse_args()
    main(args)