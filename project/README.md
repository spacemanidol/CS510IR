python -m pyserini.search --topics msmarco-passage-dev-subset    --index indexes/lucene-index-msmarco-passage  --output run.msmarco-passage.bm25tuned.txt  --bm25 --msmarco --hits 1000 --k1 0.82 --b 0.68
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir basetriples --save_steps 1000  --train_file data/train.json 
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir distill_base --teacher_model_name_or_path base/checkpoint-4000/ --save_steps 1000 
python src/rerank.py --model_name_or_path base/checkpoint-1000 --candidate_filename base1000
python src/rerank.py --model_name_or_path base/checkpoint-2000 --candidate_filename base2000
python src/rerank.py --model_name_or_path base/checkpoint-3000 --candidate_filename base3000
python src/rerank.py --model_name_or_path base/checkpoint-4000 --candidate_filename base4000
python src/rerank.py --model_name_or_path base/checkpoint-5000 --candidate_filename base5000
python src/rerank.py --model_name_or_path base/checkpoint-6000 --candidate_filename base6000

python src/ms_marco_eval.py data/qrels.dev.tsv run/base1000 >> base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/base2000 >> base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/base3000 >> base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/base4000 >> base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/base5000 >> base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/base6000 >> base_results.txt

# BERT LARGE
0.365

# Baseline 
Results:
#####################
MRR @10: 0.18741227770955546
QueriesRanked: 6980
Recal @10: 0.39856733524355303
#####################

# bert-base-uncased
#####################
MRR @10: 0.18728099356841885
QueriesRanked: 501
Recal @10: 0.3912175648702595
#####################
#####################
MRR @10: 0.196945790957767
QueriesRanked: 501
Recal @10: 0.3912175648702595
#####################
#####################
MRR @10: 0.2183506003865285
QueriesRanked: 501
Recal @10: 0.40119760479041916
#####################
#####################
MRR @10: 0.21591103507271178
QueriesRanked: 501
Recal @10: 0.4171656686626746
#####################
#####################
MRR @10: 0.20703513607705226
QueriesRanked: 501
Recal @10: 0.39520958083832336
#####################
#####################
MRR @10: 0.20766245287203372
QueriesRanked: 501
Recal @10: 0.3972055888223553
#####################
