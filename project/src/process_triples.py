import logging
import argparse
import os
import random
import sys
import numpy as np
import torch
import transformers
import pickle
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import json
def main(args):
    print("loading tokenizer")
    texts, labels = [], []
    print("Loading Data")
    with open(args.input_filename,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                texts.append((l[0],l[1]))
                labels.append(1)
                texts.append((l[0],l[2]))
                labels.append(0)
    print("Dataset Contains {} Values. Resizing to {}".format(len(texts), args.dataset_size))
    if args.dataset_size != None:
        texts = texts[:args.dataset_size]
        labels =  labels[:args.dataset_size]
    print("Spliting data into train test split")
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=args.validation_size/len(texts))
    with open('train.json', 'w') as w:
        for idx in range(len(train_texts)):
            j = {"query": train_texts[idx][0], "passage":train_texts[idx][1], "label": train_labels[idx]}
            w.write("{}\n".format(json.dumps(j)))
    with open('validation.json', 'w') as w:
        for idx in range(len(val_texts)):
            j = {"query": val_texts[idx][0], "passage":val_texts[idx][1], "label": val_labels[idx]}
            w.write("{}\n".format(json.dumps(j)))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn MSMARCO Data into HF usable datasets')
    parser.add_argument('--input_filename', type=str, default='data/triples.train.small.tsv', help='Location of triples')
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased', help='Tokenization model')
    parser.add_argument('--cache_dir', type=str, default='cache', help='cache directory')
    parser.add_argument('--dataset_size', type=int, default=None, help='Size of Training data')
    parser.add_argument('--validation_size', type=int, default=10000, help='Size of Validation set')
    parser.add_argument('--output_filename', type=str, default='msmarco_processed.pkl', help='name of processed pickle file for msmarco ourput')
    args = parser.parse_args()
    main(args)
