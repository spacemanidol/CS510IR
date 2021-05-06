python -m pyserini.search --topics msmarco-passage-dev-subset    --index indexes/lucene-index-msmarco-passage  --output run.msmarco-passage.bm25tuned.txt  --bm25 --msmarco --hits 1000 --k1 0.82 --b 0.68
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir basetriples --save_steps 1000  --train_file data/train.json 
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir distill_base --teacher_model_name_or_path base/checkpoint-4000/ --save_steps 1000 
python src/rerank.py --model_name_or_path base/checkpoint-1000 --candidate_filename base1000
python src/rerank.py --model_name_or_path base/checkpoint-2000 --candidate_filename base2000
python src/rerank.py --model_name_or_path base/checkpoint-3000 --candidate_filename base3000
python src/rerank.py --model_name_or_path base/checkpoint-4000 --candidate_filename base4000
python src/rerank.py --model_name_or_path base/checkpoint-5000 --candidate_filename base5000
python src/rerank.py --model_name_or_path base/checkpoint-6000 --candidate_filename base6000

python src/ms_marco_eval.py data/qrels.dev.tsv base1000 >> base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv base2000 >> base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv base3000 >> base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv base4000 >> base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv base5000 >> base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv base6000 >> base_results.txt



<pre>Results:
#####################
MRR @10: 0.18741227770955546
QueriesRanked: 6980
#####################
</pre> 