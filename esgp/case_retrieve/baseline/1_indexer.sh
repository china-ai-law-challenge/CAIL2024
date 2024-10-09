python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input workplace/CAIL_Task4/data/documents \
  --index workplace/CAIL_Task4/data/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw \
  --stopwords workplace/CAIL_Task4/data/others/stopword.txt