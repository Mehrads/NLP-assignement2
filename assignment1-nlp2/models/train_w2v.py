#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models import word2vec, keyedvectors, Word2Vec


def train_model(text_vec, path,name):
            model = Word2Vec(text_vec, min_count=1, window=50,epochs=5, vector_size=100, sg=1)
            model.save("{}/w2v_{}".format(path,name))
            print(f'saved w2v model ...\n {path}/w2v_{name}')

