python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qrel_20180914.txt outputs/sparse_original_query0.0.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_relevance.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qrel_20180914.txt outputs/sparse_original_query0.1.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_relevance.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qrel_20180914.txt outputs/sparse_original_query0.2.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_relevance.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qrel_20180914.txt outputs/sparse_original_query0.3.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_relevance.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qrel_20180914.txt outputs/sparse_original_query0.4.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_relevance.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qrel_20180914.txt outputs/sparse_original_query0.5.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_relevance.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qrel_20180914.txt outputs/sparse_original_query0.6.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_relevance.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qrel_20180914.txt outputs/sparse_original_query0.7.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_relevance.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qrel_20180914.txt outputs/sparse_original_query0.8.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_relevance.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qrel_20180914.txt outputs/sparse_original_query0.9.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_relevance.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qrel_20180914.txt outputs/sparse_original_query1.0.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_relevance.txt

python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qtrust_20180914.txt outputs/sparse_original_query0.0.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_auth.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qtrust_20180914.txt outputs/sparse_original_query0.1.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_auth.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qtrust_20180914.txt outputs/sparse_original_query0.2.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_auth.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qtrust_20180914.txt outputs/sparse_original_query0.3.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_auth.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qtrust_20180914.txt outputs/sparse_original_query0.4.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_auth.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qtrust_20180914.txt outputs/sparse_original_query0.5.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_auth.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qtrust_20180914.txt outputs/sparse_original_query0.6.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_auth.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qtrust_20180914.txt outputs/sparse_original_query0.7.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_auth.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qtrust_20180914.txt outputs/sparse_original_query0.8.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_auth.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qtrust_20180914.txt outputs/sparse_original_query0.9.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_auth.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qtrust_20180914.txt outputs/sparse_original_query1.0.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_auth.txt

python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qread_20180914.txt outputs/sparse_original_query0.0.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_read.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qread_20180914.txt outputs/sparse_original_query0.1.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_read.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qread_20180914.txt outputs/sparse_original_query0.2.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_read.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qread_20180914.txt outputs/sparse_original_query0.3.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_read.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qread_20180914.txt outputs/sparse_original_query0.4.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_read.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qread_20180914.txt outputs/sparse_original_query0.5.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_read.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qread_20180914.txt outputs/sparse_original_query0.6.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_read.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qread_20180914.txt outputs/sparse_original_query0.7.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_read.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qread_20180914.txt outputs/sparse_original_query0.8.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_read.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qread_20180914.txt outputs/sparse_original_query0.9.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_read.txt
python -m pyserini.eval.trec_eval -m ndcg data/CLEF2018_qread_20180914.txt outputs/sparse_original_query1.0.candidate.run | tail -n 2 | head -n 1 | awk '{print $NF}' > rm3_read.txt