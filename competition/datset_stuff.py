import ir_datasets
dataset = ir_datasets.load('msmarco-passage')
for doc in dataset.docs_iter():
    doc # namedtuple<doc_id, text>


dataset = ir_datasets.load('msmarco-passage/dev')
for query in dataset.queries_iter():
    query # namedtuple<query_id, text>

for doc in dataset.docs_iter():
    doc # namedtuple<doc_id, text>
for qrel in dataset.qrels_iter():
    qrel # namedtuple<query_id, doc_id, relevance, iteration>
for scoreddoc in dataset.scoreddocs_iter():
    scoreddoc # namedtuple<query_id, doc_id, score>


dataset = ir_datasets.load('msmarco-passage/dev/small')
for query in dataset.queries_iter():
    query # namedtuple<query_id, text>

dataset = ir_datasets.load('msmarco-passage/eval/small')
for query in dataset.queries_iter():
    query # namedtuple<query_id, text>

dataset = ir_datasets.load('msmarco-passage/train')
for query in dataset.queries_iter():
    query # namedtuple<query_id, text>
for doc in dataset.docs_iter():
    doc # namedtuple<doc_id, text>
for qrel in dataset.qrels_iter():
    qrel # namedtuple<query_id, doc_id, relevance, iteration>


dataset = ir_datasets.load('msmarco-passage/train/medical')
for query in dataset.queries_iter():
    query # namedtuple<query_id, text>

dataset = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')
for query in dataset.queries_iter():
    query # namedtuple<query_id, text>
for qrel in dataset.qrels_iter():
    qrel # namedtuple<query_id, doc_id, relevance, iteration>


dataset = ir_datasets.load('msmarco-passage/trec-dl-2020/judged')
for query in dataset.queries_iter():
    query # namedtuple<query_id, text>
for qrel in dataset.qrels_iter():
    qrel # namedtuple<query_id, doc_id, relevance, iteration>