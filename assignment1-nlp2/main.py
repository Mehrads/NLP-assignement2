#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import nltk
nltk.download('brown')
nltk.download('punkt')
from models import train_w2v
from nltk.corpus import brown
from source import golden_lists, preprocessing
from source.evaluation import evaluation, pytrec_eval_per_word, evaluation_tf_idf, pytrec_eval_perword_tfidf, pytrec_avg


def read_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize lists to store the results
    result_token1, result_token2, result_token3 = [], [], []

    for line in lines:
        tokens = line.split()
        if len(tokens) < 3:
            print(f"Skipping line due to unexpected format: {line.strip()}")
            continue  # Skip lines that don't have at least 3 tokens

        # Assuming you want the first three tokens
        rt1, rt2, rt3 = tokens[:3]
        result_token1.append(rt1)
        result_token2.append(rt2)
        result_token3.append(rt3)

    return result_token1, result_token2, result_token3



def process_and_train_w2v():
    categories = ['news', 'editorial']
    for category in categories:
        data = [preprocessing.preprocess(word) for word in brown.words(categories=category)]
        print(f'Preprocessing and training Word2Vec model for {category}...')
        train_w2v.train_model(data, "output/models/w2v_models", f"brown_{category}")

def evaluate_models(wordsList):
    models = ['brown_news', 'brown_editorial']
    for model in models:
        print(f"Evaluating and pytrec evaluation for {model}...")
        evaluation.finding_similarity(wordsList, 'output/models/w2v_models', model)
        pytrec_eval_per_word.pytrec_eval_per_word(wordsList, model)
        pytrec_avg.pytrec_avg_evaluation(wordsList, 'model', 'w2v')

def train_and_evaluate_tf_idf(resulttoken1):
    categories = ['news', 'editorial']
    for category in categories:
        data = brown.sents(categories=category)
        print(f"Training TF-IDF and evaluating for {category}...")
        evaluation_tf_idf.eval_tf_idf(data, resulttoken1, f"brown_{category}")
        pytrec_eval_perword_tfidf.pytrec_eval_per_word(wordsList, f"brown_{category}")
        pytrec_avg.pytrec_avg_evaluation(wordsList, f"brown_{category}", 'tfidf')

if __name__ == '__main__':
    dataset_path = 'data/SimLex-999.txt'
    resulttoken1, resulttoken2, resulttoken3 = read_dataset(dataset_path)

    gold_dataset_path = dataset_path
    print(f'Creating gold standard list from {gold_dataset_path}...')
    golden_lists.save_golden_lists(gold_dataset_path, 'evaluation/gold_list/gold_list.json')

    wordsList = preprocessing.remove_words(resulttoken1)
    preprocessing.gold_dataset_perword(resulttoken1)

    process_and_train_w2v()
    evaluate_models(wordsList)
    train_and_evaluate_tf_idf(resulttoken1)

