from pyserini.search import SimpleSearcher
from pyserini.dsearch import SimpleDenseSearcher, TCTColBERTQueryEncoder
from pyserini.hsearch import HybridSearcher
import xml.etree.ElementTree as ET
import os
import argparse

def load_queries(filename):
    
searcher = SimpleSearcher.from_prebuilt_index('robust04')
hits = searcher.search('hubble space telescope')

# Print the first 10 hits:
for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:15} {hits[i].score:.5f}')


searcher.set_bm25(0.9, 0.4)
searcher.set_rm3(10, 10, 0.5)

hits2 = searcher.search('hubble space telescope')

# Print the first 10 hits:
for i in range(0, 10):
    print(f'{i+1:2} {hits2[i].docid:15} {hits2[i].score:.5f}')

from pyserini.dsearch import SimpleDenseSearcher, TCTColBERTQueryEncoder

encoder = TCTColBERTQueryEncoder('castorini/tct_colbert-msmarco')
searcher = SimpleDenseSearcher.from_prebuilt_index(
    'msmarco-passage-tct_colbert-hnsw',
    encoder
)
hits = searcher.search('what is a lobster roll')

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')



ssearcher = SimpleSearcher.from_prebuilt_index('msmarco-passage')
encoder = TCTColBERTQueryEncoder('castorini/tct_colbert-msmarco')
dsearcher = SimpleDenseSearcher.from_prebuilt_index(
    'msmarco-passage-tct_colbert-hnsw',
    encoder
)
hsearcher = HybridSearcher(dsearcher, ssearcher)
hits = hsearcher.search('what is a lobster roll')

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
Evaluation measures for IRTask1: NDCG@10, BPref and RBP. trec_eval will be used for NDCG@10 and BPref: this is available for download at https://github.com/usnistgov/trec_eval.

Queries for IRTask1: https://github.com/CLEFeHealth/CLEFeHealth2018IRtask/blob/master/clef2018_queries_task1_task4.txt (Attention: only the <en> ... </en> part of the query file should be used for IRTask1)

Submission format: The format for the submission of runs should follow the standard TREC run format. Fields in the run result file should be separated using a space as the delimiter between columns. The width of the columns in the format is not important, but it is important to include all columns and have some amount of white space between the columns. Each run should contain the following fileds:

qid Q0 docno rank score tag
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sech Clef Index with various values')  
    parser.add_argument('--index_path', default='/shared/nas/data/m1/dcampos/pyserini_indexes/clef18/'
    parser.add_argument('--output_dir', default='/shared/nas/data/m1/dcampos/pyserini_indexes/clef18/')
    args = parser.parse_args()
    main(args)