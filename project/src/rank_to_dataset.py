import os
import json
import argparse

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
    

def load_qid2query(filename):
    qid2query, query2qid = {},{}
    with open(filename,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            qid2query[int(l[0])] = l[1]
            query2qid[l[1]] = int(l[0])
    return qid2query, query2qid


def main(args):
    qid2query, query2qid = load_qid2query(args.query_file)
    docid2doc, doc2docid = load_qid2query(args.collection_file)
    ranking = load_ranking(args.ranking_file, docid2doc, qid2query)
    with open(args.output_filename,'w') as w:
        for query in ranking.keys():
            for candidate in ranking[query]:
                j = {"query":query,"qid":query2qid[query], "passage":candidate, "pid": doc2docid[candidate]}
                w.write("{}\n".format(json.dumps(j)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn MSMARCO Data into HF usable datasets')
    parser.add_argument('--ranking_file', type=str, default='data/bm25devtop1000.txt', help='BM25 Ranked file')
    parser.add_argument('--query_file', type=str, default='data/queries.tsv', help="qid 2 query file")
    parser.add_argument('--collection_file', type=str, default='data/collection.tsv', help='docid 2 doc')
    parser.add_argument('--output_filename', type=str, default='devtop1000.json', help='Output file')
    args = parser.parse_args()
    main(args)
