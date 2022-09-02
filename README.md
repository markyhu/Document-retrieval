# Document-retrieval

Perform document retrieval using inverted index. The inverted index data structure is a two-level mapping:
~~~
1.{term} --> {document id} 
2.{document id} --> {term frequency in this document}
~~~

The similarity score between query and document is computed using cosine similarity. Both vectors are calculated by experimenting different term weighting modes: binary, raw term frequency, term frequency with inverse document frequency.


Evaluation is performed by comparing with the gold standard. Performance metrics with different term weighting modes and text preprocessing techniques are reported and compared.
