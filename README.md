# Document-retrieval

Perform document retrieval using inverted index. The inverted index data structure is a two-level mapping:
1.{term} --> {document id}
2.{document id} --> {term frequency in this document}

The similarity score between query and document is computed using cosine similarity. Both vectors are calculated by experimenting different term weighting mode: binary, raw term frequency, term frequency with inverse document frequency.


