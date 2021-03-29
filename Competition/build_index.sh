python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input integrations/resources/sample_collection_jsonl \
 -index/shared/nas/data/m1/dcampos3/clef_index -storePositions -storeDocvectors -storeRaw
