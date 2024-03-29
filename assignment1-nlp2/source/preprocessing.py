#!/usr/bin/env python
# coding: utf-8

# In[4]:


import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import json
nltk.download('stopwords')

# Initialize stop words once
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """
    Process input text: lowercase, tokenize, remove stopwords and non-alphabetic words.
    """
    text = text.lower()  
    tokens = word_tokenize(text)  
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words] 
    return tokens

def delete_duplicate_words(words):
    """
    Remove duplicate words from the list, preserving order.
    """
    seen = set()
    return [x for x in words if not (x in seen or seen.add(x))]

def remove_words(words):
    """
    Remove out-of-list and zero-gold-standard words based on a JSON gold list.
    """
    cleaned_words = delete_duplicate_words(words)
    final_array = []
    
    try:
        with open('evaluation/gold_list/gold_list.json') as f:
            data_gold = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return []

    for word in cleaned_words:
        if word not in ['word1'] and any(word in value for value in data_gold.values()):
            final_array.append(word)
    
    return delete_duplicate_words(final_array)

def gold_dataset_perword(words):
    """
    Create and save a gold dataset per word.
    """
    cleaned_words = delete_duplicate_words(words)
    
    try:
        with open('evaluation/gold_list/gold_list.json') as f:
            gold_data_final = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return

    for word in cleaned_words:
        if word in gold_data_final.get("golden", {}):
            sim_list = {word: []}
            gold_entries = gold_data_final["golden"][word]
            score = 10

            for entry in gold_entries:
                if entry and score > 0:  # Ensure non-empty and positive score
                    sim_list[word].append({entry: score})
                    score -= 1

            # Save to JSON
            file_path = f'output/gold_dataset_perword/gold_{word}.json'
            with open(file_path, 'w') as gold_file:
                json.dump(sim_list, gold_file, indent=4)

# Example usage (commented out to prevent accidental execution):
# words = preprocess("This is a sample text for preprocessing.")
# remove_words_result = remove_words(words)
# gold_dataset_perword(remove_words_result)

