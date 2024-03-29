#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pytrec_eval
import pandas as pd
import json
import os
from collections import OrderedDict

def read_excel_to_df(name, sample):
    """Try to read an Excel file into a DataFrame. Return None if file is empty or does not exist."""
    filepath = f'output/cosine_similarity/w2v/{name}/result_{sample}.xlsx'
    if os.path.exists(filepath) and os.stat(filepath).st_size != 0:
        return pd.read_excel(filepath, sheet_name=f'cosine_sim_{sample}')
    else:
        print(f"File {filepath} is empty or does not exist.")
        return None

def score_words(sheet, sample):
    """Score words based on the provided sheet, returning a JSON-serializable structure."""
    if sheet is not None and (sheet.values).any():
        simwords_list = {sample: []}
        for column in range(2):  # Assuming you are dealing with 2 columns based on original logic
            scores = OrderedDict((sheet.iloc[row][column], 10-row) for row in range(10))
            simwords_list[sample].append(scores)
        return simwords_list
    return None

def perform_pytrec_eval(name, sample, simwords_list):
    """Perform pytrec evaluation and save results to file."""
    filepath = f'output/gold_dataset_perword/gold_{sample}.json'
    if os.path.exists(filepath):
        with open(filepath) as gold:
            gold_data = json.load(gold)
            qrel = {key: value[0] for key, value in gold_data.items() if value}
            
            run = {sample: simwords_list[sample][0]}
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg'})
            results = evaluator.evaluate(run)
            
            output_path = f'output/pytrec/pytrec_eval_per_word/{name}/pytrec_result_perword/pytrec_{sample}.json'
            with open(output_path, 'w') as k:
                json.dump({sample: results[sample]}, k, indent=4)

def pytrec_eval_per_word(wordsList, name):
    for sample in map(str, wordsList):
        sheet = read_excel_to_df(name, sample)
        simwords_list = score_words(sheet, sample)
        if simwords_list:
            perform_pytrec_eval(name, sample, simwords_list)

# Example usage:
# pytrec_eval_per_word(['word1', 'word2'], 'modelname')

