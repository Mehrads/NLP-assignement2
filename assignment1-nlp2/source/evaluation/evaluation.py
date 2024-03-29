#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from gensim.models import Word2Vec
import xlsxwriter
from source import preprocessing


def finding_similarity(word_list, path, name):
    # Load the Word2Vec model once
    model_path = f"{path}/w2v_{name}"
    print(f'Loading Word2Vec model from {model_path}')
    model = Word2Vec.load(model_path)
    model.init_sims(replace=True)

    # Preprocess input word list
    array = preprocessing.delete_duplicate_words(word_list)
    out_of_words = []
    
    for word in array:
        # Check if the word is in the model's vocabulary
        if word not in model.wv:
            out_of_words.append(word)
            continue
        
        # Get most similar words
        similar_words = model.wv.most_similar([word])
        
        # Create Excel workbook and worksheet for each word
        workbook_path = f'output/cosine_similarity/w2v/{name}/result_{word}.xlsx'
        workbook = xlsxwriter.Workbook(workbook_path)
        worksheet = workbook.add_worksheet(f'cosine_sim_{word}')
        
        # Write header
        worksheet.write(0, 0, 'Word')
        worksheet.write(0, 1, 'Similarity')
        
        # Write similar words and their similarities
        for row, (similar_word, similarity) in enumerate(similar_words, start=1):
            worksheet.write(row, 0, similar_word)
            worksheet.write(row, 1, similarity)
        
        workbook.close()
    
    if out_of_words:
        print(f"Words not found in the model's vocabulary: {', '.join(out_of_words)}")

