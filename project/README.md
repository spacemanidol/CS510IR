python -m pyserini.search --topics msmarco-passage-dev-subset    --index indexes/lucene-index-msmarco-passage  --output run.msmarco-passage.bm25tuned.txt  --bm25 --msmarco --hits 1000 --k1 0.82 --b 0.68
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir basetriples --save_steps 1000  --train_file data/train.json 

python src/rerank.py --model_name_or_path basetriples/checkpoint-5000 --candidate_filename base5000
<pre>Results:
#####################
MRR @10: 0.18741227770955546
QueriesRanked: 6980
#####################
</pre> 