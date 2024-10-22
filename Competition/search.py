from pyserini.search import SimpleSearcher
from pyserini.dsearch import SimpleDenseSearcher, TCTColBERTQueryEncoder
from pyserini.hsearch import HybridSearcher
import xml.etree.ElementTree as ET
import os
import argparse

def load_queries(filename):
    qid2query = {}
    tree = ET.parse(filename)
    root = tree.getroot()
    for query in root:
        query_id = 0
        query_text = ''
        for child in query:
            if child.tag == 'id':
                query_id = int(child.text[1:-1])
            if child.tag == 'en':
                query_text = child.text
        if query_id != 0 and query_text != '':
            qid2query[query_id] = query_text
    return qid2query

def load_ranker(args):
    if args.sparse and args.dense:
        sparse_searcher = SimpleSearcher(args.sparse_index_path)
        sparse_searcher.set_bm25(args.k, args.b)
        sparse_searcher.set_rm3(args.expansion_terms, args.expansion_documents, args.original_query_weight)
        encoder = TCTColBERTQueryEncoder('castorini/tct_colbert-msmarco')
        dense_searcher = SimpleDenseSearcher(args.dense_index_path, encoder)
        hsearcher = HybridSearcher(dense_searcher, sparse_searcher)
    elif args.sparse:
        sparse_searcher = SimpleSearcher(args.sparse_index_path)
        sparse_searcher.set_bm25(args.k, args.b)
        sparse_searcher.set_rm3(args.expansion_terms, args.expansion_documents, args.original_query_weight)
        return sparse_searcher
    elif args.dense:
        encoder = TCTColBERTQueryEncoder('castorini/tct_colbert-msmarco')
        dense_searcher = SimpleDenseSearcher(args.dense_index_path, encoder)
        return dense_searcher
    else:
        print("Choose a valid ranking function sparse(BM25), dense(vector) or a combination of the two")
        exit(0)

def main(args):
    print("Loading Queries")
    qid2query = load_queries(args.query_file)
    print("{} Queries loaded".format(len(qid2query)))
    print("Loading Searcher")
    searcher = load_ranker(args)
    print("Searcher Loaded")
    i = 0
    with open(args.output_file, 'w') as w:
        for qid in qid2query:
            results = searcher.search(qid2query[qid])
            i += 1
            #print("{} quieries ranked".format(i))
            for i in range(0, args.topn):
                w.write("{} Q0 {} {} {} {}\n".format(qid, results[i].docid, i, results[i].score, args.tag))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sech Clef Index with various values')
    parser.add_argument('--query_file', default='data/CLEF2018queries.xml')
    parser.add_argument('--output_file', default='candidate_run')
    parser.add_argument('--topn', default=10, type=int, help='How many documents to retrieve per query')
    parser.add_argument('--k', default=1.2, type=float, help='K value for bm25')
    parser.add_argument('--b', default=0.75, type=float, help='B value for bm25')
    parser.add_argument('--dense', action='store_true', help='use dense ranking')
    parser.add_argument('--expansion_terms', default=10, type=int, help='parameter for expansion of query terms')
    parser.add_argument('--expansion_documents', default=10, type=int, help='RM3 expansion docs')
    parser.add_argument('--original_query_weight', default=1.0, type=float, help='Importance for original query')
    parser.add_argument('--sparse', action='store_true', help='use sparse ranking. Can be combined with dense')
    parser.add_argument('--tag', default="blender", type=str, help='Tag for run')
    parser.add_argument('--sparse_index_path', default='/shared/nas/data/m1/dcampos3/clef/index')
    parser.add_argument('--dense_index_path', default='/shared/nas/data/m1/dcampos3/clef/dense_index')
    args = parser.parse_args()
    main(args)
