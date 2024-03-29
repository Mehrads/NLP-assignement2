#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pytrec_eval
import pandas as pd
import json
from collections import OrderedDict
import os

def read_excel_result(name, sample, method='tfidf'):
    """Attempt to read Excel result file. Return None if file does not exist or is empty."""
    file_path = f'output/cosine_similarity/{method}/{name}/result_{sample}.xlsx'
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        print(f"File {file_path} is empty or does not exist.")
        return None
    return pd.read_excel(file_path, sheet_name=f'cosine_sim_{sample}')

def score_words_from_sheet(sheet, sample):
    """Generate scores from the sheet data."""
    if sheet is not None and sheet.values.any():
        scores = OrderedDict()
        scores[sample] = [
            OrderedDict((sheet.loc[row][0], 10 - row) for row in range(min(10, len(sheet))))
        ]
        return scores
    return None

def evaluate_and_save_results(name, sample, scores, method='tfidf'):
    """Evaluate using pytrec_eval and save the results."""
    qrel_file = f'output/gold_dataset_perword/gold_{sample}.json'
    if not os.path.exists(qrel_file):
        print(f"Qrel file {qrel_file} does not exist.")
        return

    with open(qrel_file) as f:
        qrel = json.load(f)
        # Assuming qrel structure matches expected pytrec_eval structure.

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg'})
    eval_results = evaluator.evaluate({sample: scores[sample][0]})
    
    result_file_path = f'output/pytrec/{method}_pytrec_perword/{name}/pytrec_result_perword/pytrec_{sample}.json'
    with open(result_file_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    print(f"Results saved to {result_file_path}")

def pytrec_eval_per_word(wordsList, name, method='tfidf'):
    for word in wordsList:
        sample = str(word)
        sheet = read_excel_result(name, sample, method)
        scores = score_words_from_sheet(sheet, sample)
        if scores:
            evaluate_and_save_results(name, sample, scores, method)

# Example of how to use the function:
# pytrec_eval_per_word(['exampleWord1', 'exampleWord2'], 'modelName')

