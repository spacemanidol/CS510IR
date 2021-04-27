import 

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


queries = load_qid2query('data/queries.tsv')
collection = load_qid2query('data/collection.tsv')

def main(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn MSMARCO Data into HF usable datasets')
    parser.add_argument('--ranking_file', type=str, default='data/bm25devtop1000.txt', help='BM25 Ranked fil;e')
    parser
    args = parser.parse_args()
    main(args)
