python convert_to_pyserini.py --input_dir /shared/nas/data/m1/dcampos3/clef18 --output_dir /shared/nas/data/m1/dcampos/processedclef18/
python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 8 -input /shared/nas/data/m1/dcampos3/processedclef18/  -index /shared/nas/data/m1/dcampos3/pyserini_indexes/clef -storePositions -storeDocvectors -storeRaw
