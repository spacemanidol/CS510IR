python -m pyserini.search --topics msmarco-passage-dev-subset    --index indexes/lucene-index-msmarco-passage  --output run.msmarco-passage.bm25tuned.txt  --bm25 --msmarco --hits 1000 --k1 0.82 --b 0.68

<pre>Results:
#####################
MRR @10: 0.18741227770955546
QueriesRanked: 6980
#####################
</pre>