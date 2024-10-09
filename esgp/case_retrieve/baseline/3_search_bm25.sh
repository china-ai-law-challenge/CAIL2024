python -m pyserini.search.lucene \
  --index workplace/CAIL_Task4/data/index \
  --topics workplace/CAIL_Task4/data/sample_data/queries_fact.tsv \
  --output workplace/CAIL_Task4/data/sample_data/bm25_output.tsv \
  --bm25 \
  --hits 5 \
  --threads 1 \
  --batch-size 10 \
  --stopwords workplace/CAIL/data/others/stopword.txt
 




