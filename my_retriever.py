
import math

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme.
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        
        
    # Method to compute the number of documents in the collection.
    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    
    # Method to compute the inverse document frequency of each term 
    # in the inverted index.
    def compute_idf(self):
        self.term_idf ={}
        for term in self.index:
            self.term_idf[term] = math.log(self.num_docs/len(self.index[term]))
        return self.term_idf 
    
    
    # Method to compute the term frequency in a single query.            
    
    def compute_tf_query(self,query):
        self.tf_query ={}
        for term in query:
            if term in self.tf_query:
                self.tf_query[term] +=1
            else:
                self.tf_query[term] =1
        return self.tf_query
    
        
    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        
        doc_size={} # Set empty dictionary to store document vector size
        q_d_sum={}  # Store the sum of product of query term weighting and document term weighting
        score={} #Store the similarity score
        
        if self.term_weighting =='binary':
            # Compute the document vector size under binary weighting scheme
            for term, collection in self.index.items():
                for doc_id, count in collection.items():
                    if doc_id in doc_size:
                        doc_size[doc_id] += 1
                    else:
                        doc_size[doc_id] = 1
                        
            # Query term weighting is either 1 or 0, so simply sum the number 
            # of times terms in a query appear in the document to compute q_d_sum.
            for term in query: 
                if term in self.index:
                    for doc_id in self.index[term]:
                        if doc_id in q_d_sum:
                            q_d_sum[doc_id] +=1
                        else:
                            q_d_sum[doc_id] =1
                            
         
        if self.term_weighting =='tf':
            # Compute the document vector size under tf weighting scheme.
            for term, collection in self.index.items():
                for doc_id, count in collection.items():
                    if doc_id in doc_size:
                        doc_size[doc_id] += count **2 # Square of tf
                    else:
                        doc_size[doc_id] = count **2
            
            # Compute the term frequency in the query.
            tf_query = self.compute_tf_query(query)
            
            # Compute q_d_sum under tf weighting scheme.
            for term in query:
                if term in self.index:
                    for doc_id, count in self.index[term].items():
                        if doc_id in q_d_sum:
                            q_d_sum[doc_id] += tf_query[term] * count
                        else:
                            q_d_sum[doc_id] = tf_query[term] * count
            
                
                
        if self.term_weighting =='tfidf':
            # Use the compute_idf and compute_tf_query methods defined abouve
            # to compute term inverse document frequency and query term frequency
            # respectively.
            term_idf =self.compute_idf()
            tf_query =self.compute_tf_query(query)
            
            # Compute the document vector size under tfidf weighting scheme.
            for term, collection in self.index.items():
                for doc_id, count in collection.items():
                    if doc_id in doc_size:
                        doc_size[doc_id] += (count * term_idf[term])**2 # Square of tf*idf
                    else:
                        doc_size[doc_id] = (count * term_idf[term])**2
            
            # Compute q_d_sum under tfidf weighting scheme.
            for term in query:
                if term in term_idf:
                    for doc_id, count in self.index[term].items():
                        if doc_id in q_d_sum:
                            q_d_sum[doc_id] += (tf_query[term]*term_idf[term]) * (count* term_idf[term])
                        else:
                            q_d_sum[doc_id] = (tf_query[term]*term_idf[term]) * (count* term_idf[term])
                        
                    
        # Compute the similarity score
        for doc_id in q_d_sum:
            score[doc_id] = q_d_sum[doc_id] / math.sqrt(doc_size[doc_id])
        
        # Sort the document id based on score value
        sorted_doc_ids = sorted(score, key= lambda x:score[x], reverse=True)
        
        
        return sorted_doc_ids
            
        
        
        
            
        



