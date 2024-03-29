# NLP-assignement2

# Overview 

To run this code, execute the [main.py](https://github.com/Mehrads/NLP-assignement2/blob/main/main.ipynb) in project main root directory.

**Codebases:**

 [source/](https://github.com/Mehrads/NLP-assignement2/tree/main/source) : This folder Contains code files for evaluation, gold list, preprocessing.
 
```
+---source
|   |   golden_lists.py
|   |   preprocessing.py
|   +---evaluation
|   |     evaluation.py
|   |     evaluation_tf_idf.py
|   |     pytrec_eval_per_word.py
|   |     pytrec_eval_perword_tfidf.py
|   |     pytrec_avg.py

```
[models/train_w2v.py](https://github.com/Mehrads/NLP-assignement2/blob/main/models/train_w2v.ipynb): Trains the word to vector models.

```
+---models
|   |   train_w2v.py
```

**Source Folder:**

[dataset/](https://github.com/ZahraTaherikhonakdar/Vector-vs.-Lexical-Semantics/tree/main/dataset) : This folder includes our golden standards [SimLex-9991](https://fh295.github.io/simlex.html) data set.

```
+---data
|   |   SimLex-999.txt
```
**Target Folders:**
The target folders are the outputs for the models, gold standard datasets, and cosine similarity and nDCG metric.

[evaluation/gold_list](https://github.com/Mehrads/NLP-assignement2/blob/main/source/golden_lists.ipynb) : This is the output for the ``` gold() ``` function in [main.py](https://github.com/Mehrads/NLP-assignement2/blob/main/main.ipynb) in project main root directory. It includes golden answers for each word in our golden standards [SimLex-9991](https://fh295.github.io/simlex.html) data set.

```
+---evaluation
|   +---gold_list
|   |     gold_list.json

```
[output/cosine_similarity](https://github.com/Mehrads/NLP-assignement2/tree/main/output/cosine_similarity): This is the out put for **top_k** cosine similarity for words based on word to vector and td_idf models. ``` brown_news ```,``` brown_editorial ``` folders in each ``` w2v ``` and ``` tfidf ``` folder contain the results for word_to_vectors model which is trained with [brown](https://www.nltk.org/book/ch02.html) corpus news and ditorial genres respectively. Here is the sample results for [tfidf/brown_editorial](https://github.com/Mehrads/NLP-assignement2/tree/main/output/cosine_similarity/tfidf/brown_editorial) and [w2v/brown_editorial](https://github.com/Mehrads/NLP-assignement2/tree/main/output/cosine_similarity/tfidf/brown_editorial).

```
+---output
|   +---cosine_similarity
|   |   +---tfidf     
|   |       +---brown_editorial
|   |   |   |      result_(word).xlsx
|   |       +---brown_news
|   |   |          result_(word).xlsx
|   |   +---w2v 
|   |       +---brown_editorial
|   |   |   |      result_(word).xlsx
|   |       +---brown_news
|   |   |          result_(word).xlsx

```

[output/gold_dataset_perword](https://github.com/Mehrads/NLP-assignement2/tree/main/output/gold_dataset_perword) : contains the golden dataset for each words. This is the out of calling ``` preprocess_data.gold_dataset_perwordd ``` function in  ``` eval() ```  function in [main.py](https://github.com/Mehrads/NLP-assignement2/blob/main/main.ipynb).

```
+---output
|   +---gold_dataset_perword
|   |         gold_(word).json 
```

[output/models/w2v_models](https://github.com/Mehrads/NLP-assignement2/blob/main/models/train_w2v.ipynb) : This folder keeps the word to vectors models. This is the oputput of ``` w2v() ``` function in [main.py](https://github.com/Mehrads/NLP-assignement2/blob/main/main.ipynb). 

```
+---output
|   +---models
|       +---w2v_models
|   |   |    w2v_brown_editorial 
|   |   |    w2v_brown_news
```

[output/pytrec](https://github.com/Mehrads/NLP-assignement2/tree/main/output/pytrec) : this folder contains the evaluation result for each w2v and tfidf models in ``` pytrec_eval_per_word ``` and ``` tfidf_pytrec_perword ``` respectively, which each of them contains ``` brown_editorial ``` and ``` brown_news ``` folders. In these folders the results of nDCG evaluation are stored in ``` pytrec_result_perword ``` folder. 

```
+---output
|   +---pytrec
|   |   +----pytrec_eval_per_word     
|   |       +---brown_editorial
|   |   |   |    +---pytrec_result_perword
|   |   |   |    |       pytrec_(word).json
|   |       +---brown_news
|   |   |   |    +---pytrec_result_perword
|   |   |   |    |       pytrec_(word).json
|   |       +---brown_news
|   |   |          result_(word).xlsx
|   |   +---tfidf_pytrec_perword 
|   |       +---brown_editorial
|   |   |   |    +---pytrec_result_perword
|   |   |   |    |       pytrec_(word).json
|   |       +---brown_news
|   |   |   |    +---pytrec_result_perword
|   |   |   |    |       pytrec_(word).json

```

[output/avg](https://github.com/Mehrads/NLP-assignement2/tree/main/output/avg) : Contains average nDCG results of each baseline train on two selected corpus. 

```
+---output
|   +---avg
|   |   +---tfidf     
|   |   |   ndcg_avg_brown_editorial.json
|   |   |   ndcg_avg_brown_news.json
|   |   +---w2v 
|   |   |   ndcg_avg_brown_editorial.json
|   |   |   ndcg_avg_brown_news.json

```
