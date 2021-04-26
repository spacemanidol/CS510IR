import argparse
import os
import json
import random
import sys

def load_top1000(filename):
    qid2ranking = {}
    with open(filename,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            qid = int(l[0])
            did = (l[1])
            if qid not in qid2ranking:
                qid2ranking[qid] = {}
            rank = len(qid2ranking[qid]) + 1
            qid2ranking[qid][rank] = did
    return qid2ranking

def load_reference_from_stream(f):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    qids_to_relevant_passageids = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            qid = int(l[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(int(l[2]))
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids

def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    with open(path_to_reference,'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids

def load_qid2query(filename):
    qid2query = {}
    with open(filename,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            qid2query[int(l[0])] = l[1]
    return qid2query

def main(args):
    qid2ranking = load_top1000(args.top1000_filename)
    qid2query = load_qid2query(args.query_filename)
    collection = load_qid2query(args.collection_filename)
    qrels = load_reference(args.qrel_filename)
    qids = list(qid2ranking.keys())
    random.shuffle(qids)
    qids = qids[:args.validation_size]
    with open(args.output_filename,'w') as w:
        for qid in qids:
            if qid in qrels:
                j = {"query": qid, "passage":collection[qrels[qid][0]], "label": 1}
                w.write("{}\n".format(json.dumps(j)))
                for i in range(1,args.negative_samples+1):
                    if i not in qid2ranking:
                        break
                    if qid2ranking[qid][i] != qrels[qid]:
                        j = {"query": qid, "passage":collection[int(qid2ranking[qid][i])], "label": 0}
                        w.write("{}\n".format(json.dumps(j)))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn MSMARCO Data into HF usable datasets')
    parser.add_argument('--top1000_filename', type=str, default='data/top1000.dev', help='Location of top 1000')
    parser.add_argument('--collection_filename', type=str, default='data/collection.tsv', help='Collection datafile')
    parser.add_argument('--query_filename', type=str, default='data/queries.tsv', help='query to qid file')
    parser.add_argument('--qrel_filename', type=str, default='data/qrels.dev.tsv', help='qrels file')
    parser.add_argument('--validation_size', type=int, default=300, help='Size of Validation set')
    parser.add_argument('--negative_samples', type=int, default=100, help='negative samples per query')
    parser.add_argument('--output_filename', type=str, default='evaluation.json', help='name of processed pickle file for msmarco ourput')
    args = parser.parse_args()
    main(args)
