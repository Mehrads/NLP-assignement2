#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from source import preprocessing
import xlsxwriter

def eval_tf_idf(data, word_list, name):
    unique_words = preprocessing.delete_duplicate_words(word_list)
    preprocessed_data = [" ".join(doc) for doc in data]
    print('Preprocessing training data...')

    # Initialize and fit TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectors = tfidf_vectorizer.fit_transform(preprocessed_data)

    # Create a reverse mapping from index to word
    index_to_vocab = {v: k for k, v in tfidf_vectorizer.vocabulary_.items()}

    # Evaluate section
    for word in unique_words:
        query_vector = tfidf_vectorizer.transform([word])
        cosine_similarities = cosine_similarity(query_vector, tfidf_vectors).flatten()
        top_indices = np.argsort(cosine_similarities)[-10:][::-1]

        # Create workbook and worksheet
        workbook_path = f'output/cosine_similarity/tfidf/{name}/result_(word).xlsx'
        workbook = xlsxwriter.Workbook(workbook_path)
        worksheet = workbook.add_worksheet(f'cosine_sim_{word}')

        # Write headers
        worksheet.write(0, 0, "Word")

        # Write top similar words
        for row, index in enumerate(top_indices, start=1):
            worksheet.write(row, 0, index_to_vocab[index])

        workbook.close()
        print(f'Results for "{word}" saved to "{workbook_path}".')

# Example usage
# eval_tf_idf(data=[["This is a sample document."], ["This document is another sample."]], word_list=["sample", "document"], name="example")

