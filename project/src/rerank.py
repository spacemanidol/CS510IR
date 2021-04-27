import logging
import argparse
import os
import random
import sys

import numpy as np
from datasets import load_dataset

import transformers
import torch

logger = logging.getLogger(__name__)

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

def main(args):
    print("Loading model")
    sentence1_key, sentence2_key = "query", "passage"
    padding = 'max_length'
    max_seq_length = args.max_seq_length
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    print("Model Loaded\n Loading prediction file")
    files = {"predict":args.dataset_name}
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
    training_args=TrainingArguments('.')
    training_args.per_device_eval_batch_size = args.batch_size
    trainer = Trainer(model=model,args=training_args)
    print("Dataset loaded Predicting relevance")
    predictions = trainer.predict(test_dataset=dataset["predict"]).predictions
    qid2ranking = {}
    print("Relevance Predicted. Reranking Candidates")
    for i in range(predictions.shape[0]):
        qid = idx2qidpid[i][0]
        if qid not in qid2ranking:
            qid2ranking[qid] = {}
        qid2ranking[qid][idx2qidpid[i][1]] = predictions[i][1]
    with open(args.candidate_filename, 'w') as w:
        for query in qid2ranking.keys():
            results = sorted(qid2ranking[query].items(), key = lambda kv: kv[1], reverse=True)[:args.top_n]
            for i in range(len(results)):
                w.write("{}\t{}\t{}\n".format(query, results[i][0], i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rerank Top1000')
    parser.add_argument('--model_name_or_path', type=str, default='cache/base', help="reranking model")
    parser.add_argument('--batch_size', type=int, default=512, help='eval batch size per device')
    parser.add_argument('--cache_dir', type=str, default='cache', help='cache file')
    parser.add_argument('--max_seq_length', type=int, default=256, help="padding size")
    parser.add_argument('--top_n', type=int, default=10, help="reranked results per query")
    parser.add_argument('--candidate_filename', type=str, default='candidate.tsv', help="reranked")
    parser.add_argument('--dataset_name', type=str, default='data/devtop1000.json', help='json processed results')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='bert-base-uncased', help='tokenizer')
    args = parser.parse_args()
    main(args)