python -m pyserini.search --topics msmarco-passage-dev-subset    --index indexes/lucene-index-msmarco-passage  --output run.msmarco-passage.bm25tuned.txt  --bm25 --msmarco --hits 1000 --k1 0.82 --b 0.68
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir basetriples --save_steps 1000  --train_file data/train.json 
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-1000 --candidate_filename run/base1000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-2000 --candidate_filename run/base2000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-3000 --candidate_filename run/base3000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-4000 --candidate_filename run/base4000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-5000 --candidate_filename run/base5000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-6000 --candidate_filename run/base6000
python src/ms_marco_eval.py data/qrels.dev.tsv run/base1000 >> run/base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/base2000 >> run/base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/base3000 >> run/base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/base4000 >> run/base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/base5000 >> run/base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/base6000 >> run/base_results.txt
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir models/distill_base --teacher_model_name_or_path /coldstorage/models/base/checkpoint-3000/ --save_steps 1000 --distill_hardness 1.0
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-1000 --candidate_filename run/distill_base1000
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-2000 --candidate_filename run/distill_base2000
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-3000 --candidate_filename run/distill_base3000
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-4000 --candidate_filename run/distill_base4000
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-5000 --candidate_filename run/distill_base5000
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-6000 --candidate_filename run/distill_base6000
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base1000 >> run/distill_base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base2000 >> run/distill_base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base3000 >> run/distill_base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base4000 >> run/distill_base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base5000 >> run/distill_base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base6000 >> run/distill_base_results.txt
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir models/distill_base --teacher_model_name_or_path /coldstorage/models/base/checkpoint-3000/ --save_steps 1000 --distill_hardness 0.5
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-1000 --candidate_filename run/distill_base1000
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-2000 --candidate_filename run/distill_base2000
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-3000 --candidate_filename run/distill_base3000
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-4000 --candidate_filename run/distill_base4000
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-5000 --candidate_filename run/distill_base5000
python src/rerank.py --model_name_or_path models/distill_base/checkpoint-6000 --candidate_filename run/distill_base6000
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base1000 >> run/distill_base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base2000 >> run/distill_base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base3000 >> run/distill_base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base4000 >> run/distill_base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base5000 >> run/distill_base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/distill_base6000 >> run/distill_base_results.txt


python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir 9layer --save_steps 1000  --train_file data/train.json --layers_to_keep 9
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-1000 --candidate_filename run/9base1000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-2000 --candidate_filename run/9base2000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-3000 --candidate_filename run/9base3000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-4000 --candidate_filename run/9base4000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-5000 --candidate_filename run/9base5000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-6000 --candidate_filename run/9base6000
python src/ms_marco_eval.py data/qrels.dev.tsv run/9base1000 >> run/9base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/9base2000 >> run/9base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/9base3000 >> run/9base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/9base4000 >> run/9base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/9base5000 >> run/9base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/9base6000 >> run/9base_results.txt

python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir 6layer --save_steps 1000  --train_file data/train.json --layers_to_keep 6
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-1000 --candidate_filename run/6base1000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-2000 --candidate_filename run/6base2000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-3000 --candidate_filename run/6base3000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-4000 --candidate_filename run/6base4000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-5000 --candidate_filename run/6base5000
python src/rerank.py --model_name_or_path /coldstorage/models/base/checkpoint-6000 --candidate_filename run/6base6000
python src/ms_marco_eval.py data/qrels.dev.tsv run/6base1000 >> run/6base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/6base2000 >> run/6base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/6base3000 >> run/6base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/6base4000 >> run/6base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/6base5000 >> run/6base_results.txt
python src/ms_marco_eval.py data/qrels.dev.tsv run/6base6000 >> run/6base_results.txt


python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/80sparse --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/80sparse.yaml
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/80sparse-distill --teacher_model_name_or_path /coldstorage/models/base/checkpoint-3000  --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/80sparse.yaml
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/80sparse6layer --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/80sparse6layer.yaml
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/80sparse-distill-6layer --teacher_model_name_or_path /coldstorage/models/base/checkpoint-3000  --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/80sparse6layer.yaml --layers_to_keep 6
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/90sparse --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/90sparse.yaml
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/90sparse-distill --teacher_model_name_or_path /coldstorage/models/base/checkpoint-3000  --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/90sparse.yaml
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/90sparse6layer --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/90sparse6layer.yaml
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/90sparse-distill-6layer --teacher_model_name_or_path /coldstorage/models/base/checkpoint-3000  --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/90sparse6layer.yaml --layers_to_keep 6
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/97sparse --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/97sparse.yaml
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/97sparse-distill --teacher_model_name_or_path /coldstorage/models/base/checkpoint-3000  --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/97sparse.yaml
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/97sparse6layer --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/97sparse6layer.yaml
python src/train.py --per_device_train_batch_size 64 --fp16 --output_dir /coldstorage/models/97sparse-distill-6layer --teacher_model_name_or_path /coldstorage/models/base/checkpoint-3000  --save_steps 1000  --train_file data/train.json --nm_prune_config recipes/97sparse6layer.yaml --layers_to_keep 6

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
