#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
from source import preprocessing

def create_golden_lists(words, relatedwords, scores):
    """
    Creates golden lists from given word pairs and their scores, eliminating duplicates.
    """
    processed_words = preprocessing.delete_duplicate_words(words)
    data = {'golden': []}

    for word in processed_words:
        related_words_set = set()
        for k, current_word in enumerate(words):
            if word == current_word and float(scores[k]) > 0:
                related_words_set.update(
                    related_word for i, related_word in enumerate(relatedwords) if related_word == words[i] and float(scores[i]) > 0
                )

        # Add to golden list after removing duplicates among related words
        data['golden'].append({
            word: list(preprocess_data.delete_duplicate_words(list(related_words_set)))
        })

    return data

def save_golden_lists(input_file, output_file):
    """
    Reads an input file to generate and save golden lists to the specified output file.
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    token1, token2, scores = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:  # Ensure there are enough parts to avoid IndexError
            token1.append(parts[0])
            token2.append(parts[1])
            scores.append(parts[3])

    data = create_golden_lists(token1, token2, scores)

    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

    print(f'Saved golden lists to: {output_file}')

# Example usage (commented out to prevent accidental execution):
# save_golden_lists('input_file.txt', 'output_file.json')

